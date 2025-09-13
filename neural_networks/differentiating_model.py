from logging import info
from time import time
from typing import Any, Tuple, Union
import keras
import numpy as np
import tensorflow as tf

from data_tools.data_utils import DataSet
from data_tools.detector.constants import TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import save_training_outcomes
from train.train_config import TrainConfig


class DifferentiatingModel(keras.models.Model):
    def __init__(
        self,
        context: ExecutionContext,
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
        self.detector_deltas = tf.Variable(
            trainable=True,
            initial_value=np.zeros(shape=tuple(self._config.detector__binning_number_of_bins)),
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

    def call(self, data_set: DataSet) -> np.ndarray:
        return super().call(data_set)


def calc_t_LFVNN(
        context: ExecutionContext,
        sample_dataset: DataSet,
        reference_dataset: DataSet,
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
    model = DifferentiatingModel(context, name=name)
    model.compile(loss=model.ddp_symmetrized_loss,  optimizer='adam')
    tau_model_fit = model.fit(
        np.array(feature_dataset.events, dtype=np.float32),
        np.array(target_structure, dtype=np.float32),
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
