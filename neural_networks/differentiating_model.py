from __future__ import annotations

from contextlib import contextmanager
from logging import info
from time import time
from typing import Any, List, Tuple, Union
import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from data_tools.detector.detector_effect import DetectorEffect
from data_tools.data_utils import DataSet
from data_tools.detector.constants import TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_structure import TENSORBOARD_LOG_FILE_NAME
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
        self._context = context
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
        self._bins_of_events = None  # Set in context
        self.detector_deltas = {self._observable_names[i]: tf.Variable(
            trainable=True,
            initial_value=np.zeros(shape=(nbins,), dtype=np.float32),
            dtype="float32",
            name=f"detector-binwise-nuisances-{i}",
        ) for i, nbins in enumerate(detector_effect._numbers_of_bins)}

    @staticmethod
    def __single_nuisance_loss(nuisance_value: Any):
        std = tf.cast(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD, tf.float32)
        return tf.exp(-0.5 * tf.square(nuisance_value / std)) / (std * tf.sqrt(2.0 * tf.cast(np.pi, tf.float32)))

    def __observable_nuisance_loss(self, observable_name: str) -> float:
        return tf.reduce_prod(self.__single_nuisance_loss(
            self.detector_deltas[observable_name]
        ))

    def _nuisance_aux_loss(self) -> float:
        return tf.reduce_prod([
            self.__observable_nuisance_loss(obs) for obs in self._observable_names
        ])
    
    def _prediction_loss(
            self,
            f__is_sample_prediction: np.ndarray,
            y__is_sample_truth: np.ndarray,
        ) -> float:
        is_ref_truth = 1 - y__is_sample_truth
        return tf.reduce_sum(
            is_ref_truth * (tf.exp(f__is_sample_prediction) - 1) \
                - y__is_sample_truth * f__is_sample_prediction
        )

    @property
    def _observable_names(self) -> List[str]:
        return self._detector_effect._observable_names

    def ddp_symmetrized_loss(
            self,
            y__is_sample_truth,
            f__is_sample_prediction,
        ) -> float:
        """
        Symmetrized DDP custom loss for optimizing likelihood of the
        estimation.
        """
        return self._prediction_loss(
            f__is_sample_prediction,
            y__is_sample_truth,
        ) - self._nuisance_aux_loss()

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

    def get_metrics(self) -> List[tf.keras.metrics.Metric]:
        class PredictionLossMetric(keras.metrics.Metric):
            def update_state(self, y_true, y_pred, sample_weight=None):
                self.y_true, self.y_pred = y_true, y_pred
            
            def result(inner_self):
                return self._prediction_loss(inner_self.y_true, inner_self.y_pred)

        class NuisanceAuxLossMetric(keras.metrics.Metric):
            def update_state(self, y_true, y_pred, sample_weight=None):
                pass
            
            def result(inner_self):
                return self._nuisance_aux_loss()
        
        class SingleNuisanceLossMetric(NuisanceAuxLossMetric):
            def result(self):
                return super().result() / sum(len(vars) for vars in self.detector_deltas)

        class NuisanceAbsSumMetric(keras.metrics.Metric):
            def update_state(inner_self, y_true, y_pred, sample_weight=None):
                pass
            
            def result(inner_self):
                return tf.reduce_sum(tf.stack([
                    tf.reduce_sum(tf.abs(var)) for var in self.detector_deltas.values()
                ]))

        return [
            PredictionLossMetric(name=HistoryKeys.PREDICTION_LOSS.value),
            NuisanceAuxLossMetric(name=HistoryKeys.NUISANCE_LOSS.value),
            SingleNuisanceLossMetric(name=HistoryKeys.SINGLE_NUISANCE_LOSS.value),
            NuisanceAbsSumMetric(name=HistoryKeys.NUISANCE_ABS_SUM.value),
        ]

    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        return [
            keras.callbacks.TensorBoard(
                log_dir=self._context.training_outcomes_dir / TENSORBOARD_LOG_FILE_NAME, # type: ignore
                histogram_freq=self._config.train__number_of_epochs_for_checkpoint,
            ),
        ]

    def fit(self, data: DataSet, target: npt.NDArray, **kwargs):
        """
        Overload of the fit method to be used with DataSet objects and one-time calculation of binning.

        batch_size is hardcoded for the slicing should be done along with the data slicing, and this is not implemented.
        """
        with self.binning_context(data):
            return super().fit(data.events, y=target, batch_size=data.n_samples, **kwargs)
        
    def predict(self, data: DataSet, **kwargs) -> npt.NDArray:
        """
        Overload of the predict method to be used with DataSet objects and one-time calculation of binning.

        batch_size is hardcoded for the slicing should be done along with the data slicing, and this is not implemented.
        """
        with self.binning_context(data):
            return super().predict(data.events, batch_size=data.n_samples, **kwargs)

    def call(self, data_set: tf.Tensor) -> tf.Tensor:
        naive_prediction = super().call(data_set)

        # Each event weight is multiplied by the exponentiation multiplication of all affecting nuisances
        nuisance_skews = [
            tf.gather(tf.exp(self.detector_deltas[obs]), self._bins_of_events[:, i])
            for i, obs in enumerate(self._observable_names)
        ]

        items = tf.stack([tf.squeeze(naive_prediction), *nuisance_skews])
        return tf.reduce_prod(items, axis=0)


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
    model.compile(
        loss=model.ddp_symmetrized_loss,
        metrics=model.get_metrics(),
        optimizer='adam',
    )
    tau_model_fit = model.fit(
        feature_dataset,
        target_structure,
        sample_weight=loss_weights,
        epochs=context.config.train__epochs,
        verbose=0,
        callbacks=model.get_callbacks(),
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
