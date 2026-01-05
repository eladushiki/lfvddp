from __future__ import annotations

from contextlib import contextmanager
import gc
from logging import info, debug, warning
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
from neural_networks.utils import MAX_PREDICTION_CUTOFF, MIN_PREDICTION_CUTOFF, save_training_outcomes
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

        # Initialize weights according to strategy
        self._initialize_weights()

        # Logging setup
        self._setup_tensorboard()

    def _build_layers(self):
        self._input_layer = keras.Input(shape=(self._config.train__nn_input_dimension,))
        last_layer = self._input_layer
        for i, secondary_layer_size in enumerate(self._config.train__nn_architecture[1:]):
            # Create layers with default initializers; actual initialization done in _initialize_weights()
            if i == len(self._config.train__nn_architecture[1:]) - 1:
                layer = keras.layers.Dense(
                    secondary_layer_size,
                    activation=None,  # No activation on final layer
                )(last_layer)
            else:
                layer = keras.layers.Dense(
                    secondary_layer_size,
                    activation=keras.layers.LeakyReLU(negative_slope=0.01),
                )(last_layer)
            last_layer = layer
        self._last_layer = last_layer

    def _build_detector_nuisances(self):
        self._detector_deltas = {}
        for i, nbins in enumerate(self._detector_effect._numbers_of_bins):
            if self._config.train__data_is_train_for_nuisances:
                nuisance_var = self.add_weight(
                    name=f"detector-bin-nuisances-{i}",
                    shape=(nbins,),
                    dtype=tf.float32,
                    trainable=True,
                )
            else:
                nuisance_var = tf.Variable(
                    initial_value=tf.zeros(shape=(nbins,)),
                    dtype=tf.float32,
                    trainable=False,
                )
            self._detector_deltas[self._observable_names[i]] = nuisance_var

    def _create_initial_weights(self) -> List[tf.Tensor]:
        """
        Create newly initialized weights matching the training strategy.
        This is the single source of truth for weight initialization.
        """
        new_weights = []
        for layer in self.layers[1:]:
            # Kernel: Use HeNormal for hidden layers, GlorotNormal for output layer
            initializer = keras.initializers.HeNormal() if layer != self.layers[-1] else keras.initializers.GlorotNormal()
            new_weights.append(
                initializer(layer.kernel.shape, dtype=layer.kernel.dtype)
            )
            # Bias: Use GlorotNormal
            new_weights.append(
                keras.initializers.GlorotNormal()(layer.bias.shape, dtype=layer.bias.dtype)
            )

        # Handle detector nuisances separately
        if self._config.train__data_is_train_for_nuisances:
            for var in self._detector_deltas.values():
                new_weights.append(
                    keras.initializers.RandomNormal(
                        mean=0.0,
                        stddev=float(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD),
                    )(var.shape, dtype=var.dtype)
                )

        return new_weights

    def _initialize_weights(self):
        """Initialize weights using the centralized strategy."""
        self.set_weights(self._create_initial_weights())

    def _setup_tensorboard(self):
        # Initialize
        tf.keras.backend.clear_session()
        gc.collect()

        # Create logging directory
        self._current_epoch = tf.Variable(0, trainable=False, dtype=tf.int64)
        self.tensorboard_log_dir = self._context.training_outcomes_dir / TENSORBOARD_LOG_DIR_NAME / self.name
        self._train_summary_writer = tf.summary.create_file_writer(str(self.tensorboard_log_dir))  # type: ignore
        
    @tf.function
    def _gaussian_nuisance_nll(self, nuisance_value: Any) -> tf.Tensor:
        """
        Negative log-likelihood of a single (or vector of) nuisance parameter(s) under
        Gaussian constraint.
        - log(x) = 0.5 * (x/σ)² + log(σ√(2π))
        Constant term is dropped.
        return: tf.Tensor: Tensor of same shape as nuisance_value with NLL values.
        """
        std = tf.cast(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD, tf.float32)
        return 0.5 * tf.square(nuisance_value / std)

    @tf.function
    def _total_nuisance_nll(self) -> tf.Tensor:
        """
        Total negative log-likelihood for all nuisance parameters, summed over observables.
        Calculated directly as a sum after taking the individual NLLs.
        return: tf.Tensor: Scalar tensor of total nuisance NLL.
        """
        nuisances = tf.concat([tf.reshape(var, [-1]) for var in self._detector_deltas.values()], axis=0)
        return tf.reduce_sum(self._gaussian_nuisance_nll(nuisances))
    
    @tf.function
    def _prediction_nll(
            self,
            y__is_sample_truth: tf.Tensor,
            f__is_sample_pred: tf.Tensor,
        ) -> tf.Tensor:
        """
        The custom negative log-likelihood for the prediction of the NN.
        Rewards correct classification of sample vs. reference events.
        return: tf.Tensor: Tensor of same shape as input tensors with NLL values.
        """
        is_ref_truth = tf.subtract(1.0, y__is_sample_truth)
        return is_ref_truth * (tf.exp(f__is_sample_pred) - 1) \
            - tf.multiply(y__is_sample_truth, f__is_sample_pred)

    @property
    def _observable_names(self) -> List[str]:
        return self._detector_effect._observable_names

    @tf.function
    def ddp_symmetrized_loss(
            self,
            y__is_sample_truth: tf.Tensor,
            f__is_sample_pred: tf.Tensor,
        ) -> tf.Tensor:
        """
        Symmetrized DDP custom loss for optimizing likelihood of the
        estimation. Returns negative log-likelihood to be minimized.
        return: tf.Tensor: Tensor of same shape as input tensors with total NLL values.
        tf automatically reweights the loss by sample_weight given in fit(), as long
        as this function returns a tf.Tensor of shape (batch_size,).
        """
        prediction_loss = self._prediction_nll(
            y__is_sample_truth=y__is_sample_truth,
            f__is_sample_pred=f__is_sample_pred,
        )  # Tensor the size of data
        if self._config.train__data_is_train_for_nuisances:  # todo: this division by sample size makes no sense. Remove
            nuisance_loss = self._total_nuisance_nll()  # Scalar
        else:
            nuisance_loss = 0.0

        # Total loss is sum of log-likelihoods. Addition by tf is element-wise.
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
                val = tf.reduce_sum(self._prediction_nll(y_pred, y_true))
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
            PredictionLossMetric(name=HistoryKeys.PREDICTION_LOSS.value),
            NuisanceNegLogLikelihoodMetric(name=HistoryKeys.NUISANCE_LOSS.value),
        ] + ([NuisanceAbsSumMetric(name=HistoryKeys.NUISANCE_ABS_SUM.value)] if self._config.train__data_is_train_for_nuisances else [])

    
    class ExtremePredictionResetCallback(keras.callbacks.Callback):
        """Reset weights if model gets stuck at maximum or minimum predicted values."""
        sample_data = None

        def __init__(
                self,
                max_stuck_epochs=50,
                max_threshold=MAX_PREDICTION_CUTOFF - 0.5,
                min_threshold=MIN_PREDICTION_CUTOFF + 0.5,
                stuck_fraction_threshold=0.5,
                check_interval=100,
        ):
            super().__init__()
            self._max_stuck_epochs = max_stuck_epochs
            self._max_threshold = max_threshold
            self._min_threshold = min_threshold
            self._stuck_fraction_threshold = stuck_fraction_threshold
            self._consecutive_stuck_epochs = 0
            self._initial_weights = None
            self._check_interval = check_interval

        def on_epoch_end(self, epoch, logs=None):
            # Only check every check_interval epochs
            if epoch % self._check_interval != 0:
                return
            
            # Get predictions on actual data (without the symbolic tensor issue)
            sample_predictions = self.model(self.sample_data, training=False)
            
            # Ensure predictions are numpy for comparison
            if hasattr(sample_predictions, 'numpy'):
                sample_predictions = sample_predictions.numpy()
            
            # Check if most predictions are stuck at max or min
            stuck_at_max = float(tf.reduce_mean(
                tf.cast(sample_predictions >= self._max_threshold, tf.float32)
            ).numpy())
            stuck_at_min = float(tf.reduce_mean(
                tf.cast(sample_predictions <= self._min_threshold, tf.float32)
            ).numpy())
            
            is_stuck = (stuck_at_max >= self._stuck_fraction_threshold) or \
                        (stuck_at_min >= self._stuck_fraction_threshold)
            stuck_type = "max" if stuck_at_max >= stuck_at_min else "min"
            stuck_fraction = max(stuck_at_max, stuck_at_min)
            
            if is_stuck:
                self._consecutive_stuck_epochs += self._check_interval
                info(f"Epoch {epoch}: {stuck_fraction:.2%} of predictions stuck at {stuck_type}. " +
                        f"Consecutive stuck epochs: {self._consecutive_stuck_epochs}/{self._max_stuck_epochs}")
                
                if self._consecutive_stuck_epochs >= self._max_stuck_epochs:
                    info(f"Resetting model weights after {self._consecutive_stuck_epochs} consecutive stuck epochs")
                    self._consecutive_stuck_epochs = 0
                    self.model._initialize_weights()
            else:
                if self._consecutive_stuck_epochs > 0:
                    info(f"Epoch {epoch}: Model recovered. Resetting stuck epoch counter.")
                self._consecutive_stuck_epochs = 0
 
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
                ]) if self._config.train__data_is_train_for_nuisances else "No nuisance parameters"
                log_text = inner_self.TEXT_LOG_TEMPLATE.format(
                    epoch=epoch,
                    nuisance_values=nuisance_values,
                )
                with self._train_summary_writer.as_default():
                    tf.summary.text("nuisance_parameters", log_text, step=epoch)

                self._train_summary_writer.flush()
        
        class EarlyStoppingDetectorCallback(keras.callbacks.Callback):
            """Detects if training was interrupted (e.g., by walltime or OOM)."""
            def on_train_begin(inner_self, logs=None):
                self._current_epoch.assign(0)
            
            def on_epoch_end(inner_self, epoch, logs=None):
                self._current_epoch.assign(epoch)
                if epoch % (self._config.train__number_of_epochs_for_checkpoint * 5) == 0:
                    debug(f'Completed epoch {epoch}/{self._config.train__epochs}')
                    if logs:
                        loss_val = logs.get('loss', 'N/A')
                        debug(f'  Loss: {loss_val}')
                        if isinstance(loss_val, float) and (np.isnan(loss_val) or np.isinf(loss_val)):
                            warning(f'Loss is {loss_val} at epoch {epoch} - training may diverge')
               
        return [
            keras.callbacks.TensorBoard(
                log_dir=self.tensorboard_log_dir, # type: ignore
                histogram_freq=self._config.train__number_of_epochs_for_checkpoint,
                update_freq='epoch',
            ),
            TextLoggerCallback(),
            EarlyStoppingDetectorCallback(),
            self.ExtremePredictionResetCallback(),
        ]

    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]):
        """
        A custom loop is implemented in order to learn the nuisance variables as well as the model's weights.
        """
        x, y, weights = keras.utils.unpack_x_y_sample_weight(data)

        # Record operations while calling the NN for auto differentiation
        with tf.GradientTape() as tape:

            prediction = self(data=x, training=True)
            tf.debugging.assert_all_finite(prediction, message="Prediction contains NaN or Inf values")

            loss = self.compute_loss(
                x=x, y=y,
                y_pred=prediction,
                sample_weight=weights,
            )
            tf.debugging.assert_all_finite(loss, message="Loss contains NaN or Inf values")

        # Use tape to update trainable vars. Apply a single step
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics and return results with explicit loss tracking.
        for metric in self.metrics:
            if metric.name == HistoryKeys.LOSS.value:
                metric.update_state(loss)
            else:
                metric.update_state(y, prediction, sample_weight=weights)
                
        return {m.name: m.result() for m in self.metrics}

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
            self.ExtremePredictionResetCallback.sample_data = data.events
            return super().fit(x=data.events, y=target, batch_size=self._config.train__batch_size, **kwargs)

    def predict(self, data: DataSet, **kwargs) -> npt.NDArray:
        """
        Overload of the predict method to be used with DataSet objects and one-time calculation of binning.

        batch_size is hardcoded for the slicing should be done along with the data slicing, and this is not implemented.
        """
        with self.binning_context(data):
            return super().predict(x=data.events, batch_size=data.n_samples, **kwargs)

    @tf.function
    def call(self, data: tf.Tensor, training: bool = None) -> tf.Tensor:
        naive_prediction = super().call(data, training=training)
        
        # Clip naive prediction to prevent overflow in exp() during loss calculation
        # lower: due to underflow
        # upper: theoretically the correct reweight is no more than exp(0)
        safe_prediction = tf.clip_by_value(naive_prediction, MIN_PREDICTION_CUTOFF, MAX_PREDICTION_CUTOFF)

        # Each event weight is multiplied by the exponentiation multiplication of all affecting nuisances
        if self._config.train__data_is_train_for_nuisances:
            nuisance_skews = [
                tf.gather(tf.exp(self._detector_deltas[obs]), self._bins_of_events[:, i])
                for i, obs in enumerate(self._observable_names)
            ]

            items = tf.stack([tf.squeeze(safe_prediction), *nuisance_skews])
            return tf.reduce_prod(items, axis=0)
        else:
            return tf.squeeze(safe_prediction)


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
            sample_dataset._weight_mask,
            reference_dataset._weight_mask * sample_dataset.corrected_n_samples / reference_dataset.corrected_n_samples,
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
    # Use Adam optimizer with L2 regularization to improve convergence
    optimizer = keras.optimizers.Adam()
    model.compile(
        loss=model.ddp_symmetrized_loss,
        metrics=model.get_metrics(),
        optimizer=optimizer,
    )
    tau_model_fit = model.fit(
        data=feature_dataset,
        target=target_structure,
        sample_weight=loss_weights,
        epochs=context.config.train__epochs,
        verbose=0,
        callbacks=model.get_callbacks(),
    )
    tau_model_history = tau_model_fit.history
    
    # Log diagnostic information about training completion
    info(f'Total epochs requested: {context.config.train__epochs}')
    info(f'Actual epochs completed: {len(tau_model_history[HistoryKeys.LOSS.value])}')
    info(f'Early stopping occurred: {len(tau_model_history[HistoryKeys.LOSS.value]) < context.config.train__epochs}')
    
    # Check for NaN/Inf in loss history
    loss_array = np.array(tau_model_history[HistoryKeys.LOSS.value])
    if np.any(np.isnan(loss_array)):
        nan_idx = np.where(np.isnan(loss_array))[0]
        info(f'NaN detected in loss at epochs: {nan_idx}')
    if np.any(np.isinf(loss_array)):
        inf_idx = np.where(np.isinf(loss_array))[0]
        info(f'Inf detected in loss at epochs: {inf_idx}')
    
    tau_model_history[HistoryKeys.EPOCH.value] = np.concatenate([
        np.arange(0, context.config.train__epochs, context.config.train__number_of_epochs_for_checkpoint),
        np.array([context.config.train__epochs - 1]),
    ])
    tau_history = np.array(tau_model_history[HistoryKeys.LOSS.value])[tau_model_history[HistoryKeys.EPOCH.value]]
    tau_model_history[HistoryKeys.LOSS.value] = tau_history

    info(f'Training time (seconds): {time() - t0}')

    # Keras automatically averages the loss over samples by dividing by sum(sample_weight).
    # We need to rescale to get the unaveraged loss for the correct t-statistic.
    total_weight = np.sum(loss_weights)
    final_loss_unaveraged = tau_history[-1] * total_weight
    final_loss = calc_t_test_statistic(final_loss_unaveraged)
    info(f'Observed t test statistic: {final_loss}')
    
    save_training_outcomes(
        context,
        model_history=tau_model_history,
        tau_model=model,
    )

    return model, final_loss
