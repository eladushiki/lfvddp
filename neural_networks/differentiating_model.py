from __future__ import annotations

from contextlib import contextmanager
from logging import info, debug, warning
from pathlib import Path
from time import time
from typing import Dict, List, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from data_tools.detector.detector_effect import DetectorEffect
from data_tools.data_utils import DataSet
from data_tools.detector.constants import TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_structure import TENSORBOARD_LOG_DIR_NAME
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import MAX_PREDICTION_CUTOFF, MIN_PREDICTION_CUTOFF, ContextedModel, save_training_outcomes, get_model_logging_dir
from train.train_config import TrainConfig


class DifferentiatingModel(pl.LightningModule, ContextedModel):
    """
    Symmetrized DDP's model used to estimate the test statistic using PyTorch Lightning.
    A custom loss function is used to find the maximizing parameters for hypothesis.
    """
    def __init__(
        self,
        context: ExecutionContext,
        detector_effect: DetectorEffect,
        name: str,
        **kwargs
    ):
        super().__init__()
        self._name = name
        self._context = context
        self._config: Union[TrainConfig, DetectorConfig] = context.config

        # Add layers by spec
        self._build_layers()
        
        # Add detector uncertainty nuisance parameters
        self._detector_effect = detector_effect
        self._build_detector_nuisances()
        self._bins_of_events = None  # Set in context

        # Initialize NN parameters according to strategy
        self._initialize_parameters()
        
        # Store training data and weights for metrics computation
        self._train_data = None
        self._train_target = None
        self._train_weights = None
        self._training_history = {}
        
        # Track minimum loss for checkpointing
        self._min_loss = float('inf')
        self._best_model_path = None

    def _build_layers(self):
        # Fully connected 2-layer network:
        input_dim = self._config.train__nn_input_dimension
        hidden_size = self._config.train__nn_inner_layer_nodes
        output_size = self._config.train__nn_output_dimension
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(hidden_size, output_size),
        )

    def _build_detector_nuisances(self):
        self._detector_deltas = {}
        for i, nbins in enumerate(self._detector_effect._numbers_of_bins):
            if self._config.train__data_is_train_for_nuisances:
                nuisance_var = nn.Parameter(
                    torch.zeros(nbins, dtype=torch.float32, device=self.device)
                )
            else:
                nuisance_var = torch.zeros(nbins, dtype=torch.float32, device=self.device)
            self._detector_deltas[self._observable_names[i]] = nuisance_var
        
        # Register parameters if trainable
        if self._config.train__data_is_train_for_nuisances:
            for name, var in self._detector_deltas.items():
                self.register_parameter(f"nuisance_{name}", var)

    def _create_initial_parameters(self) -> None:
        """
        Create newly initialized weights matching the training strategy.
        This is the single source of truth for weight initialization.
        Assumes 2-layer network (1 hidden layer).
        """
        # Use Xavier uniform with reduced gain parameter (0.5) for smaller initial values
        gain = 0.1
        
        # Layer 0: Input to hidden
        hidden_layer = self.network[0]
        nn.init.xavier_uniform_(hidden_layer.weight, gain=gain)
        nn.init.xavier_uniform_(hidden_layer.bias.view(-1, 1), gain=gain)
        
        # Layer 2: Hidden to output (skipping LeakyReLU at index 1)
        output_layer = self.network[2]
        nn.init.xavier_uniform_(output_layer.weight, gain=gain)
        nn.init.xavier_uniform_(output_layer.bias.view(-1, 1), gain=gain)

        # Handle detector nuisances separately
        if self._config.train__data_is_train_for_nuisances:
            for var in self._detector_deltas.values():
                nn.init.normal_(var, mean=0.0, std=float(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD))

    def _initialize_parameters(self):
        """Initialize parameters using the centralized strategy."""
        self._create_initial_parameters()

    def configure_optimizers(self):
        """Configure optimizer for Lightning."""
        return optim.Adam(self.parameters())

    def _gaussian_nuisance_nll(self, nuisance_value: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood of a single (or vector of) nuisance parameter(s) under
        Gaussian constraint.
        - log(x) = 0.5 * (x/σ)² + log(σ√(2π))
        Constant term is dropped.
        return: torch.Tensor: Tensor of same shape as nuisance_value with NLL values.
        """
        std = torch.tensor(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD, dtype=torch.float32, device=self.device)
        return 0.5 * torch.square(nuisance_value / std)

    def _total_nuisance_nll(self) -> torch.Tensor:
        """
        Total negative log-likelihood for all nuisance parameters, summed over observables.
        Calculated directly as a sum after taking the individual NLLs.
        return: torch.Tensor: Scalar tensor of total nuisance NLL.
        """
        nuisances = torch.cat([var.reshape(-1) for var in self._detector_deltas.values()])
        return torch.sum(self._gaussian_nuisance_nll(nuisances))
    
    def _prediction_nll(
            self,
            y__is_sample_truth: torch.Tensor,
            f__is_sample_pred: torch.Tensor,
        ) -> torch.Tensor:
        """
        The custom negative log-likelihood for the prediction of the NN.
        Rewards correct classification of sample vs. reference events.
        return: torch.Tensor: Tensor of same shape as input tensors with NLL values.
        """
        is_ref_truth = 1.0 - y__is_sample_truth
        return is_ref_truth * (torch.exp(f__is_sample_pred) - 1) \
            - y__is_sample_truth * f__is_sample_pred

    @property
    def _observable_names(self) -> List[str]:
        return self._detector_effect._observable_names

    def ddp_symmetrized_loss(
            self,
            y__is_sample_truth: torch.Tensor,
            f__is_sample_pred: torch.Tensor,
        ) -> torch.Tensor:
        """
        Symmetrized DDP custom loss for optimizing likelihood of the
        estimation. Returns negative log-likelihood to be minimized.
        return: torch.Tensor: Tensor of same shape as input tensors with total NLL values.
        """
        prediction_loss = self._prediction_nll(
            y__is_sample_truth=y__is_sample_truth,
            f__is_sample_pred=f__is_sample_pred,
        )  # Tensor the size of data
        if self._config.train__data_is_train_for_nuisances:
            nuisance_loss = self._total_nuisance_nll()  # Scalar
        else:
            nuisance_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Total loss is sum of log-likelihoods
        return prediction_loss + nuisance_loss

    def forward(self, data: torch.Tensor, training: bool = True) -> torch.Tensor:
        naive_prediction = self.network(data)
        
        # Clip naive prediction to prevent overflow in exp() during loss calculation
        safe_prediction = torch.clamp(naive_prediction, MIN_PREDICTION_CUTOFF, MAX_PREDICTION_CUTOFF)

        # Each event predicted weight is multiplied by the exponentiation multiplication of all affecting nuisances
        if self._config.train__data_is_train_for_nuisances:
            nuisance_skews = [
                torch.gather(torch.exp(self._detector_deltas[obs]), 0, self._bins_of_events[:, i])
                for i, obs in enumerate(self._observable_names)
            ]

            items = torch.stack([safe_prediction.squeeze(), *nuisance_skews])
            return torch.prod(items, dim=0)
        else:
            return safe_prediction.squeeze()

    def _on_epoch_end(self, epoch: int, logs: Dict[str, float] = None):
        """Log metrics for end of epoch."""
        if logs is None:
            logs = {}
        
        # Log scalar metrics with Lightning
        for metric_name, metric_value in logs.items():
            self.log(metric_name, metric_value, prog_bar=True)
        
        # Log nuisance parameters as text if trainable
        if self._config.train__data_is_train_for_nuisances:
            nuisance_values = "\n".join([
                f"{name}: {var.detach().cpu().numpy()}"
                for name, var in self._detector_deltas.items()
            ])
            log_text = f"Nuisance parameters at epoch {epoch}:\n{nuisance_values}"
            self.logger.experiment.add_text("nuisance_parameters", log_text, epoch)
        
        # Log diagnostic information
        if epoch % (self._config.train__number_of_epochs_for_checkpoint * 5) == 0:
            debug(f'Completed epoch {epoch}/{self._config.train__epochs}')
            if 'loss' in logs:
                loss_val = logs.get('loss')
                debug(f'  Loss: {loss_val}')
                if isinstance(loss_val, float) and (np.isnan(loss_val) or np.isinf(loss_val)):
                    warning(f'Loss is {loss_val} at epoch {epoch} - training may diverge')

    class ExtremePredictionResetCallback(Callback):
        """Reset NN if model gets stuck at maximum or minimum predicted values."""

        def __init__(
                self,
                max_stuck_epochs: int = 50,
                max_threshold: float = MAX_PREDICTION_CUTOFF - 0.5,
                min_threshold: float = MIN_PREDICTION_CUTOFF + 0.5,
                stuck_fraction_threshold: float = 0.5,
                check_interval: int = 100,
        ):
            super().__init__()
            self._max_stuck_epochs = max_stuck_epochs
            self._max_threshold = max_threshold
            self._min_threshold = min_threshold
            self._stuck_fraction_threshold = stuck_fraction_threshold
            self._consecutive_stuck_epochs = 0
            self._check_interval = check_interval

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: 'DifferentiatingModel') -> None:
            """Called at the end of each training epoch."""
            epoch = trainer.current_epoch
            
            # Only check every check_interval epochs
            if epoch % self._check_interval != 0:
                return
            
            # Get predictions on training data
            with torch.no_grad():
                sample_predictions = pl_module(pl_module._train_data)
                sample_predictions = sample_predictions.detach().cpu().numpy()
            
            # Check if most predictions are stuck at max or min
            stuck_at_max = float(np.mean(sample_predictions >= self._max_threshold))
            stuck_at_min = float(np.mean(sample_predictions <= self._min_threshold))
            
            is_stuck = (stuck_at_max >= self._stuck_fraction_threshold) or \
                        (stuck_at_min >= self._stuck_fraction_threshold)
            stuck_type = "max" if stuck_at_max >= stuck_at_min else "min"
            stuck_fraction = max(stuck_at_max, stuck_at_min)
            
            if is_stuck:
                self._consecutive_stuck_epochs += self._check_interval
                info(f"Epoch {epoch}: {stuck_fraction:.2%} of predictions stuck at {stuck_type}. " +
                        f"Consecutive stuck epochs: {self._consecutive_stuck_epochs}/{self._max_stuck_epochs}")
                
                if self._consecutive_stuck_epochs >= self._max_stuck_epochs:
                    warning(f"Resetting model parameters after {self._consecutive_stuck_epochs} consecutive stuck epochs")
                    self._consecutive_stuck_epochs = 0
                    pl_module._initialize_parameters()
            else:
                if self._consecutive_stuck_epochs > 0:
                    info(f"Epoch {epoch}: Model recovered. Resetting stuck epoch counter.")
                self._consecutive_stuck_epochs = 0

    class BestLossCheckpointCallback(Callback):
        """Save model checkpoint whenever a lower weighted loss is achieved."""

        def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: 'DifferentiatingModel') -> None:
            """Called at the end of each training epoch."""
            current_loss = trainer.callback_metrics.get('loss')
            
            if current_loss is not None:
                current_loss = float(current_loss)
                
                if current_loss < pl_module._min_loss:
                    # Delete previous best checkpoint if it exists
                    if pl_module._best_model_path is not None:
                        Path(pl_module._best_model_path).unlink()
                        debug(f"Deleted previous best checkpoint: {pl_module._best_model_path}")
                    
                    pl_module._min_loss = current_loss
                    # Save new best checkpoint
                    checkpoint_path = pl_module._get_best_checkpoint_path(trainer.current_epoch)
                    trainer.save_checkpoint(checkpoint_path)
                    pl_module._best_model_path = checkpoint_path
                    info(f"New best loss {current_loss:.6f} at epoch {trainer.current_epoch}. "
                         f"Checkpoint saved: {checkpoint_path}")

    def _get_best_checkpoint_path(self, epoch: int) -> str:
        """Generate path for best model checkpoint."""
        tensorboard_log_dir = get_model_logging_dir(self._context, self._name)
        return str(tensorboard_log_dir / f"best_model_epoch_{epoch}.ckpt")

    @contextmanager
    def binning_context(self, data: DataSet):
        try:
            self._bins_of_events = torch.tensor(
                self._detector_effect.get_event_bin_centers(data, indexed=True),
                dtype=torch.long,
                device=self.device
            )
            yield
        finally:
            self._bins_of_events = None

    def _prepare_training_data(
        self,
        data: DataSet,
        target: npt.NDArray,
        weights: npt.NDArray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert DataSet and target to tensors on the correct device."""
        x_tensor = torch.tensor(data.events, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(target, dtype=torch.float32, device=self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return x_tensor, y_tensor, weights_tensor

    def _create_data_loader(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
    ) -> DataLoader:
        """Create a DataLoader for the training data."""
        batch_size = x_tensor.shape[0] if self._config.train__batch_size is None else self._config.train__batch_size
        dataset = TensorDataset(x_tensor, y_tensor, weights_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        batch_x, batch_y, batch_weights = batch

        # Forward pass
        predictions = self(batch_x, training=True)

        # Check for NaN/Inf
        if torch.any(~torch.isfinite(predictions)):
            raise ValueError(f"Prediction contains NaN or Inf values on epoch {self.current_epoch}")

        # Compute loss
        batch_loss = self.ddp_symmetrized_loss(
            batch_y,
            predictions,
        )
        
        # Apply sample weights
        weighted_loss = (batch_loss * batch_weights).mean()
        self.log("loss", weighted_loss, prog_bar=True)
        return weighted_loss
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch - compute full metrics."""
        with torch.no_grad():
            full_predictions = self(self._train_data, training=False)
            metrics = self._calculate_metrics(self._train_target, full_predictions)
        
        logs = {'loss': self.trainer.callback_metrics.get('loss', 0.0)}
        logs.update(metrics)
        self._on_epoch_end(self.current_epoch, logs)
        
        # Store in training history
        for key, value in logs.items():
            if key not in self._training_history:
                self._training_history[key] = []
            self._training_history[key].append(value)

    def _calculate_metrics(
        self,
        y__is_sample_truth: torch.Tensor,
        f__is_sample_pred: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calculate all metrics in a structured manner.
        Returns a dict with all relevant metric values.
        """
        metrics = {}
        
        # Prediction loss
        prediction_nll = self._prediction_nll(y__is_sample_truth, f__is_sample_pred)
        metrics[HistoryKeys.PREDICTION_LOSS.value] = torch.sum(prediction_nll).item()
        
        # Nuisance loss and absolute sum
        if self._config.train__data_is_train_for_nuisances:
            metrics[HistoryKeys.NUISANCE_LOSS.value] = self._total_nuisance_nll().item()
            metrics[HistoryKeys.NUISANCE_ABS_SUM.value] = sum(
                torch.sum(torch.abs(var)).item() for var in self._detector_deltas.values()
            )
        else:
            metrics[HistoryKeys.NUISANCE_LOSS.value] = 0.0
        
        return metrics

    def _get_callbacks(self) -> List[Callback]:
        """Get callbacks list, creating defaults if none provided."""
        return [
            self.ExtremePredictionResetCallback(
                max_stuck_epochs=50,
                max_threshold=MAX_PREDICTION_CUTOFF - 0.5,
                min_threshold=MIN_PREDICTION_CUTOFF + 0.5,
                stuck_fraction_threshold=0.5,
                check_interval=100,
            ),
            self.BestLossCheckpointCallback(),
        ]

    def fit(
        self,
        data: DataSet,
        target: npt.NDArray,
        weights: npt.NDArray,
    ) -> Dict[str, List[float]]:
        """
        Training loop using PyTorch Lightning.
        
        Args:
            data: Training DataSet object
            target: Target labels array
            weights: Sample weight array
            callbacks: Optional list of Lightning callbacks
        
        Returns:
            Dictionary with training history
        """
        # Store training data for metrics computation
        self._train_data, self._train_target, self._train_weights = self._prepare_training_data(data, target, weights)

        # Prepare data loader
        dataloader = self._create_data_loader(self._train_data, self._train_target, self._train_weights)

        # Get callbacks
        callbacks_list = self._get_callbacks()

        # Set up tensorboard logger
        tensorboard_log_dir = get_model_logging_dir(self._context, self._name)
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        trainer = pl.Trainer(
            max_epochs=self._config.train__epochs,
            callbacks=callbacks_list,
            logger=pl.loggers.TensorBoardLogger(
                save_dir=str(tensorboard_log_dir.parent),
                name=self._name,
            ),
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        # Training with binning context
        with self.binning_context(data):
            trainer.fit(self, dataloader)

        # Collect history from training
        return self._training_history

    def predict(self, data: DataSet) -> npt.NDArray:
        """
        Prediction method to be used with DataSet objects and one-time calculation of binning.
        """
        with self.binning_context(data):
            x_tensor = torch.tensor(data.events, dtype=torch.float32, device=self.device)
            self.eval()
            with torch.no_grad():
                predictions = self(x_tensor, training=False)
            return predictions.cpu().numpy()

    def save_parameters(self, file_path) -> None:
        """Save PyTorch model parameters to file."""
        torch.save(self.state_dict(), file_path)

    def load_best_checkpoint(self) -> None:
        """Load the best checkpoint found during training if one exists."""
        if self._best_model_path is not None:
            info(f"Loading best model from checkpoint: {self._best_model_path}")
            checkpoint = torch.load(self._best_model_path, map_location=self.device, weights_only=False)
            self.load_state_dict(checkpoint['state_dict'])
        else:
            warning("No best checkpoint found. Using current model state.")


def calc_t_LFVNN(
        context: ExecutionContext,
        sample_dataset: DataSet,
        reference_dataset: DataSet,
        detector_effect: DetectorEffect,
        name: str,
) -> Tuple[pl.LightningModule, float]:
    
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
    
    tau_model = DifferentiatingModel(
        context=context,
        detector_effect=detector_effect,
        name=name,
    )

    tau_model_history = tau_model.fit(
        data=feature_dataset,
        target=target_structure,
        weights=loss_weights,
    )
    
    info(f'Training time (seconds): {time() - t0}')
    
    # Load the best checkpoint found during training
    tau_model.load_best_checkpoint()

    # PyTorch Lightning loss is already the sum over batch samples.
    # We need to rescale to get the correct t-statistic using the minimum loss achieved.
    total_weight = np.sum(loss_weights)
    min_loss = tau_model._min_loss * total_weight
    final_test_statistic = calc_t_test_statistic(min_loss)
    info(f'Minimum weighted loss achieved: {min_loss:.6f}')
    info(f'Observed t test statistic: {final_test_statistic}')
    
    save_training_outcomes(
        context,
        model_history=tau_model_history,
        tau_model=tau_model,
    )

    return tau_model, final_test_statistic
