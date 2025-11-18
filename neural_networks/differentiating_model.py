from __future__ import annotations

from contextlib import contextmanager
import gc
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
from frame.file_structure import TENSORBOARD_LOG_DIR_NAME
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
        self._build_layers()
        
        # Build model
        super(DifferentiatingModel, self).__init__(
            name=name,
            inputs=self._input_layer,
            outputs=self._last_layer,
            **kwargs
        )

        # Add detector uncertainty nuisance parameters
        self._detector_effect = detector_effect
        self._build_detector_nuisances()
        self._bins_of_events = None  # Set in context

        # Logging setup
        self._setup_tensorboard()

    def _build_layers(self):
        self._input_layer = keras.Input(shape=(self._config.train__nn_input_dimension,))
        last_layer = self._input_layer
        for i, secondary_layer_size in enumerate(self._config.train__nn_architecture[1:]):
            # Use a small positive bias initializer for the output layer to avoid zero initialization
            if i == len(self._config.train__nn_architecture[1:]) - 1:
                # Final layer: initialize bias to small positive value to avoid all-zero outputs
                layer = keras.layers.Dense(
                    secondary_layer_size,
                    activation=None,  # No activation on final layer
                    bias_initializer=keras.initializers.Constant(0.1),
                    kernel_initializer='glorot_uniform',
                )(last_layer)
            else:
                layer = keras.layers.Dense(
                    secondary_layer_size,
                    activation='relu',
                )(last_layer)
            last_layer = layer
        self._last_layer = last_layer

    def _build_detector_nuisances(self):
        self._detector_deltas = {}
        for i, nbins in enumerate(self._detector_effect._numbers_of_bins):
            nuisance_var = self.add_weight(
                name=f"detector-bin-nuisances-{i}",
                shape=(nbins,),
                dtype=tf.float32,
                trainable=True,
                initializer=keras.initializers.RandomNormal(
                    mean=0.0,
                    stddev=float(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD),
                ),
            )
            self._detector_deltas[self._observable_names[i]] = nuisance_var

    def _setup_tensorboard(self):
        # Initialize
        tf.keras.backend.clear_session()
        gc.collect()

        # Create logging directory
        self._current_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._tensorboard_log_file = self._context.training_outcomes_dir / TENSORBOARD_LOG_DIR_NAME / self.name
        self._train_summary_writer = tf.summary.create_file_writer(str(self._tensorboard_log_file))  # type: ignore
        
    @tf.function
    def _gaussian_nuisance_nll(self, nuisance_value: Any) -> tf.Tensor:
        """
        Negative log-likelihood of a single nuisance parameter under Gaussian constraint.
        - log(x) = 0.5 * (x/σ)² + log(σ√(2π))
        We can drop the constant term for optimization purposes
        """
        std = tf.cast(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD, tf.float32)
        return 0.5 * tf.square(nuisance_value / std)

    @tf.function
    def _total_nuisance_nll(self) -> tf.Tensor:
        """
        Total negative log-likelihood for all nuisance parameters.
        Calculated directly as a sum after taking the individual NLLs.
        """
        nuisances = tf.concat([tf.reshape(var, [-1]) for var in self._detector_deltas.values()], axis=0)
        tf.print("Nuisance values:", nuisances)
        return tf.reduce_sum(self._gaussian_nuisance_nll(nuisances))
    
    @tf.function
    def _prediction_nll(
            self,
            f__is_sample_prediction: tf.Tensor,
            y__is_sample_mask: tf.Tensor,
        ) -> tf.Tensor:
        is_ref_mask = tf.subtract(1.0, y__is_sample_mask)
        return is_ref_mask * (tf.exp(f__is_sample_prediction) - 1) \
            - tf.multiply(y__is_sample_mask, f__is_sample_prediction)

    @property
    def _observable_names(self) -> List[str]:
        return self._detector_effect._observable_names

    @tf.function
    def ddp_symmetrized_loss(
            self,
            y__is_sample_truth: tf.Tensor,
            f__is_sample_prediction: tf.Tensor,
        ) -> tf.Tensor:
        """
        Symmetrized DDP custom loss for optimizing likelihood of the
        estimation. Returns negative log-likelihood to be minimized.
        """
        prediction_loss = self._prediction_nll(
            f__is_sample_prediction,
            y__is_sample_truth,
        )  # Tensor the size of data
        nuisance_loss = self._total_nuisance_nll() / tf.cast(tf.shape(y__is_sample_truth)[0], tf.float32)  # Scalar

        # Total loss is sum of log-likelihoods
        return tf.math.add(prediction_loss, nuisance_loss)

    def get_metrics(self) -> List[tf.keras.metrics.Metric]:

        class NuisanceAbsSumMetric(keras.metrics.Metric):
            def __init__(inner_self, **kwargs):
                super().__init__(**kwargs)
                inner_self.__value = inner_self.add_weight(name="nuisance_abs_sum", initializer="zeros")

            def update_state(inner_self, y_true, y_pred, sample_weight=None):
                val = tf.reduce_sum(tf.stack([
                    tf.reduce_sum(tf.abs(var)) for var in self._detector_deltas.values()
                ]))
                inner_self.__value.assign(val)
            
            def result(inner_self):
                return inner_self.__value
            
            def reset_state(inner_self):
                inner_self.__value.assign(0.0)

        class PredictionLossMetric(keras.metrics.Metric):
            def __init__(inner_self, **kwargs):
                super().__init__(**kwargs)
                inner_self.__value = inner_self.add_weight(name="prediction_loss", initializer="zeros")

            def update_state(inner_self, y_true, y_pred, sample_weight=None):
                batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
                val = tf.reduce_sum(self._prediction_nll(y_pred, y_true)) / batch_size
                inner_self.__value.assign(val)

            def result(inner_self):
                return inner_self.__value

            def reset_state(inner_self):
                inner_self.__value.assign(0.0)

        class NuisanceNegLogLikelihoodMetric(keras.metrics.Metric):
            def __init__(inner_self, **kwargs):
                super().__init__(**kwargs)
                inner_self.__value = inner_self.add_weight(name="nuisance_loss", initializer="zeros")

            def update_state(inner_self, y_true, y_pred, sample_weight=None):
                nuisance_loss = self._total_nuisance_nll()
                inner_self.__value.assign(nuisance_loss)
            
            def result(inner_self):
                return inner_self.__value
            
            def reset_state(inner_self):
                inner_self.__value.assign(0.0)

        return [
            NuisanceAbsSumMetric(name=HistoryKeys.NUISANCE_ABS_SUM.value),
            PredictionLossMetric(name=HistoryKeys.PREDICTION_LOSS.value),
            NuisanceNegLogLikelihoodMetric(name=HistoryKeys.NUISANCE_LOSS.value),
        ]

    def get_callbacks(self) -> List[keras.callbacks.Callback]:
        class TextLoggerCallback(keras.callbacks.Callback):
            TEXT_LOG_TEMPLATE = f"""
            Nuisance parameters at epoch {{epoch}}:
            {{nuisance_values}}
            """
            def on_epoch_end(inner_self, epoch, logs=None):
                nuisance_values = "\n".join([
                    f"{name}: {var.numpy()}"
                    for name, var in self._detector_deltas.items()
                ])
                log_text = inner_self.TEXT_LOG_TEMPLATE.format(
                    epoch=epoch,
                    nuisance_values=nuisance_values,
                )
                with self._train_summary_writer.as_default():
                    tf.summary.text("nuisance_parameters", log_text, step=epoch)

                self._train_summary_writer.flush()
                
        return [
            keras.callbacks.TensorBoard(
                log_dir=self._tensorboard_log_file, # type: ignore
                histogram_freq=self._config.train__number_of_epochs_for_checkpoint,
                update_freq='epoch',
            ),
            TextLoggerCallback(),
        ]

    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        """
        A custom loop is implemented in order to learn the nuisance variables as well as the model's weights.
        """
        x, y, weights = keras.utils.unpack_x_y_sample_weight(data)

        # Record operations while calling the NN for auto differentiation
        with tf.GradientTape() as tape:
            prediction = self(x, training=True)
            loss = self.compute_loss(
                x, y,
                prediction,
                sample_weight=weights,
                training=True,
            )

        # Use tape to update trainable vars. Apply a single step
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics via the compiled Keras utilities (handles sample weights)
        return self.compute_metrics(x, y, y_pred=prediction, sample_weight=weights)

    @contextmanager
    def binning_context(self, data: DataSet):
        try:
            self._bins_of_events = self._detector_effect.get_event_bin_centers(data, indexed=True)
            yield
        finally:
            self._bins_of_events = None

    def fit(self, data: DataSet, target: npt.NDArray, **kwargs) -> keras.callbacks.History:
        """
        Overload of the fit method to be used with DataSet objects and one-time calculation of binning.

        batch_size is hardcoded for the slicing should be done along with the data slicing, and this is not implemented.
        """
        with self.binning_context(data):
            return super().fit(data.events, target, batch_size=data.n_samples, **kwargs)

    def predict(self, data: DataSet, **kwargs) -> npt.NDArray:
        """
        Overload of the predict method to be used with DataSet objects and one-time calculation of binning.

        batch_size is hardcoded for the slicing should be done along with the data slicing, and this is not implemented.
        """
        with self.binning_context(data):
            return super().predict(data.events, batch_size=data.n_samples, **kwargs)

    def call(self, data_set: tf.Tensor) -> tf.Tensor:
        naive_prediction = super().call(data_set)
        
        # Clip naive prediction to prevent overflow in exp() during loss calculation
        # Max value of ~20 keeps exp(20) ≈ 485 million, which is large but manageable
        # To allow for gradient flow, use stop_gradient trick
        clipped_naive_prediction = tf.clip_by_value(naive_prediction, -20.0, 20.0)
        safe_prediction = tf.math.add(naive_prediction, tf.stop_gradient(clipped_naive_prediction - naive_prediction))

        # Each event weight is multiplied by the exponentiation multiplication of all affecting nuisances
        nuisance_skews = [
            tf.gather(tf.exp(self._detector_deltas[obs]), self._bins_of_events[:, i])
            for i, obs in enumerate(self._observable_names)
        ]

        items = tf.stack([tf.squeeze(safe_prediction), *nuisance_skews])
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
    model = DifferentiatingModel(
        context,
        detector_effect,
        name=name,
    )
    # Use gradient clipping to prevent exploding gradients
    optimizer = keras.optimizers.Adam(clipnorm=1.0)
    model.compile(
        loss=model.ddp_symmetrized_loss,
        metrics=model.get_metrics(),
        optimizer=optimizer,
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
