from __future__ import annotations

from contextlib import contextmanager
from logging import info
from time import time
from typing import Any, Tuple, Union
import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from data_tools.detector_effect import DetectorEffect
from data_tools.data_utils import DataSet
from data_tools.detector.constants import TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import save_training_outcomes
from train.train_config import TrainConfig


class DifferentiatingModel(keras.models.Model):
    """
    Symmetrized DDP's model used to estimate the test statistic.
    A custom loss function is used to find the maximizing parameters for hypothesis.
    """
    def __init__(
        self,
        context: ExecutionContext,
        detector_effect: DetectorEffect,
        name: str,
        **kwargs
    ):
        self._config: Union[TrainConfig, DetectorConfig] = context.config

        # Add layers by spec
        input_layer = keras.Input(shape=(self._config.train__nn_input_dimension,))
        last_layer = input_layer
        for secondary_layer_size in self._config.train__nn_architecture[1:]:
            layer = keras.layers.Dense(
                secondary_layer_size,
                activation='relu',
            )(last_layer)
            last_layer = layer
        
        # Build model
        super(DifferentiatingModel, self).__init__(
            name=name,
            inputs=input_layer,
            outputs=last_layer,
            **kwargs
        )

        # Add detector uncertainty nuisance parameters
        self._detector_effect = detector_effect
        self.detector_deltas = tf.Variable(
            trainable=True,
            initial_value=np.zeros(shape=tuple(detector_effect._numbers_of_bins)),
            dtype="float32",
            name="detector-binwise-nuisances",
        )

    @staticmethod
    def __single_nuisance_loss(nuisance_value: Any):
        std = tf.cast(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD, tf.float32)
        return tf.exp(-0.5 * tf.square(nuisance_value / std)) / (std * tf.sqrt(2.0 * tf.cast(np.pi, tf.float32)))

    def __nuisance_aux_loss(self) -> float:
        return tf.reduce_prod(
            self.__single_nuisance_loss(
                self.detector_deltas
            )
        )
    
    def __prediction_loss(
            self,
            f__is_sample_prediction: np.ndarray,
            y__is_sample_truth: np.ndarray,
        ) -> float:
        is_ref_truth = 1 - y__is_sample_truth
        return tf.reduce_sum(
            is_ref_truth * (tf.exp(f__is_sample_prediction) - 1) \
                - y__is_sample_truth * f__is_sample_prediction
        )
    
    def ddp_symmetrized_loss(
            self,
            y__is_sample_truth,
            f__is_sample_prediction,
        ) -> float:
        """
        Symmetrized DDP custom loss for optimizing likelihood of the
        estimation.
        """
        return self.__prediction_loss(
            f__is_sample_prediction,
            y__is_sample_truth,
        ) - self.__nuisance_aux_loss()

    @contextmanager
    def binning_context(self, data: DataSet):
        """
        Context is necessary each time a new dataset is used.
        This is implemented this way to only run once the binning calculation.
        """
        try:
            self._bins_of_events = self._detector_effect.get_event_bin_centers(data, indexed=True)
            yield
        finally:
            self._bins_of_events = None

    def fit(self, data: DataSet, target: npt.NDArray, **kwargs):
        with self.binning_context(data):
            return super().fit(data.events, y=target, **kwargs)
        
    def predict(self, data: DataSet, **kwargs) -> npt.NDArray:
        with self.binning_context(data):
            return super().predict(data.events, **kwargs)

    def call(self, data_set: tf.Tensor) -> tf.Tensor:
        naive_prediction = super().call(data_set)

        # Each event weight is multiplied by the exponentiation multiplication of all affecting nuisances
        nuisance_skews = tf.gather_nd(tf.exp(self.detector_deltas), self._bins_of_events)

        return tf.multiply(naive_prediction, nuisance_skews)


def calc_t_LFVNN(
        context: ExecutionContext,
        sample_dataset: DataSet,
        reference_dataset: DataSet,
        detector_effect: DetectorEffect,
        name: str,
) -> Tuple[keras.models.Model, float]:
    
    ## Done preparing sample
    feature_dataset = sample_dataset + reference_dataset
    target_structure = np.concatenate((
            np.ones(shape=(sample_dataset.n_samples,)),
            np.zeros(shape=(reference_dataset.n_samples,)),
        ),
        axis=0,
    )
    loss_weights = np.concatenate((
            sample_dataset._weight_mask / sample_dataset.corrected_n_samples,
            reference_dataset._weight_mask / reference_dataset.corrected_n_samples,
        ),
        axis=0,
    )

    # Train
    info("Starting training")
    t0 = time()
    
    # Just fit without any special training, like is done in LFVNN
    model = DifferentiatingModel(context, detector_effect, name=name)
    model.compile(loss=model.ddp_symmetrized_loss,  optimizer='adam')
    tau_model_fit = model.fit(
        feature_dataset,
        target_structure,
        sample_weight=loss_weights,
        epochs=context.config.train__epochs,
        batch_size=feature_dataset.n_samples,
        verbose=0,
    )
    tau_model_history = tau_model_fit.history
    tau_model_history[HistoryKeys.EPOCH.value] = np.concatenate([
        np.arange(0, context.config.train__epochs, context.config.train__number_of_epochs_for_checkpoint),
        np.array([context.config.train__epochs - 1]),
    ])
    tau_history = np.array(tau_model_history[HistoryKeys.LOSS.value])[tau_model_history[HistoryKeys.EPOCH.value]]
    tau_model_history[HistoryKeys.LOSS.value] = tau_history

    info(f'Training time (seconds): {time() - t0}')

    final_loss = calc_t_test_statistic(tau_history[-1])
    info(f'Observed t test statistic: {final_loss}')
    
    save_training_outcomes(
        context,
        model_history=tau_model_history,
        tau_model=model,
    )

    return model, final_loss
