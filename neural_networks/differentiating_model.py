from __future__ import annotations

from contextlib import contextmanager
from logging import info, warning
from time import time
from typing import Dict, List, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

from data_tools.detector.detector_effect import DetectorEffect
from data_tools.data_utils import DataSet
from data_tools.detector.constants import TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import ContextedModel, save_training_outcomes, get_model_logging_dir
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
        self._create_initial_parameters()
        
        # Store training data and weights for metrics computation
        self._train_data = None
        self._train_target_classifier = None
        self._train_weights = None
        self._norm_factor = None
        self._training_history = {}

    def _build_layers(self):
        # Fully connected 2-layer network:
        input_dim = self._config.train__nn_input_dimension
        hidden_size = self._config.train__nn_inner_layer_nodes
        output_size = self._config.train__nn_output_dimension
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
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
        # Use Xavier uniform with configurable gain for weight initialization
        gain = self._config.train__nn_xavier_gain
        
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

    def configure_optimizers(self):
        """Configure optimizer for Lightning with learning rate scheduling."""
        # LBFGS optimizer for better convergence on smooth loss landscapes
        optimizer = optim.LBFGS(
            self.parameters(),
            lr=self._config.train__learning_rate,
            max_iter=20,  # Max iterations per step
            line_search_fn='strong_wolfe',  # Line search strategy
        )
        
        # No learning rate scheduler needed for LBFGS
        return optimizer

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
            is_sample_classifier: torch.Tensor,
            e_to_the_f_prediction: torch.Tensor,
        ) -> torch.Tensor:
        """
        The custom negative log-likelihood for the prediction of the NN.
        Rewards correct classification of sample vs. reference events.
        return: torch.Tensor: Tensor of same shape as input tensors with NLL values.
        """
        is_ref_classifier = 1.0 - is_sample_classifier
        return is_ref_classifier * (e_to_the_f_prediction - 1) \
            - is_sample_classifier * torch.log(e_to_the_f_prediction)

    @property
    def _observable_names(self) -> List[str]:
        return self._detector_effect._observable_names

    def ddp_symmetrized_loss(
            self,
            is_sample_classifier: torch.Tensor,
            e_to_the_f_prediction: torch.Tensor,
        ) -> torch.Tensor:
        """
        Symmetrized DDP custom loss for optimizing likelihood of the
        estimation. Returns negative log-likelihood to be minimized.
        return: torch.Tensor: Tensor of same shape as input tensors with total NLL values.
        """
        prediction_loss = self._prediction_nll(
            is_sample_classifier=is_sample_classifier,
            e_to_the_f_prediction=e_to_the_f_prediction,
        )  # Tensor the size of data
        if self._config.train__data_is_train_for_nuisances:
            nuisance_loss = self._total_nuisance_nll()  # Scalar
        else:
            nuisance_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # Total loss is sum of log-likelihoods
        return prediction_loss + nuisance_loss

    def forward(self, data: torch.Tensor, training: bool = True) -> torch.Tensor:
        e_to_the_f_prediction = self.network(data)
        
        # Each event predicted weight is multiplied by the exponentiation multiplication of all affecting nuisances
        if self._config.train__data_is_train_for_nuisances:
            nuisance_skews = [
                torch.gather(torch.exp(self._detector_deltas[obs]), 0, self._bins_of_events[:, i])
                for i, obs in enumerate(self._observable_names)
            ]

            items = torch.stack([e_to_the_f_prediction.squeeze(), *nuisance_skews])
            return torch.sum(items, dim=0)
        else:
            return e_to_the_f_prediction.squeeze()

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
        target_classifier: npt.NDArray,
        weights: npt.NDArray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert DataSet and target to tensors on the correct device."""
        data_tensor = torch.tensor(data.events, dtype=torch.float32, device=self.device)
        target_classifier_tensor = torch.tensor(target_classifier, dtype=torch.float32, device=self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        return data_tensor, target_classifier_tensor, weights_tensor

    def _create_data_loader(
        self,
        x_tensor: torch.Tensor,
        y_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
        batch_size: int,
    ) -> DataLoader:
        dataset = TensorDataset(x_tensor, y_tensor, weights_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Lightning training step."""
        batch_x, batch_y, batch_weights = batch

        # Forward pass
        e_to_the_f_predictions = self(batch_x, training=True)

        # Compute per-sample loss
        per_sample_loss = self.ddp_symmetrized_loss(
            batch_y,
            e_to_the_f_predictions,
        )
        
        # Apply weights and aggregate to scalar
        weighted_loss = per_sample_loss * batch_weights
        return torch.mean(weighted_loss)
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch - compute full metrics."""
        with torch.no_grad():
            e_to_the_f_prediction = self(self._train_data, training=False)
            logs = self._calculate_metrics(
                self._train_target_classifier,
                e_to_the_f_prediction,
            )
        
        # Log scalar metrics with Lightning
        for metric_name, metric_value in logs.items():
            self.log(metric_name, metric_value, prog_bar=True)
        
        # Store in training history (include epoch number)
        for key, value in logs.items():
            if key not in self._training_history:
                self._training_history[key] = []
            self._training_history[key].append(value)

    def _calculate_metrics(
        self,
        is_sample_classifier: torch.Tensor,
        e_to_the_f_prediction: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Calculate all metrics in a structured manner.
        Returns a dict with all relevant metric values.
        """
        metrics = {}
        metrics[HistoryKeys.EPOCH.value] = self.current_epoch

        # Prediction loss
        prediction_nll = self._prediction_nll(
            is_sample_classifier,
            e_to_the_f_prediction,
        )
        weighted_prediction_nll = prediction_nll * self._train_weights
        metrics[HistoryKeys.MEAN_LOSS.value] = torch.mean(weighted_prediction_nll).item()
        metrics[HistoryKeys.LOSS.value] = torch.sum(weighted_prediction_nll).item()

        # Nuisance loss and absolute sum
        if self._config.train__data_is_train_for_nuisances:
            metrics[HistoryKeys.NUISANCE_LOSS.value] = self._total_nuisance_nll().item()
            metrics[HistoryKeys.NUISANCE_ABS_SUM.value] = sum(
                torch.sum(torch.abs(var)).item() for var in self._detector_deltas.values()
            )
        else:
            metrics[HistoryKeys.NUISANCE_LOSS.value] = 0.0
            metrics[HistoryKeys.NUISANCE_ABS_SUM.value] = 0.0
        
        return metrics

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
        
        Returns:
            Dictionary with training history
        """
        # Store training data for metrics computation
        normalized_data, self._norm_factor = data.get_normalized()
        self._train_data, self._train_target_classifier, self._train_weights = self._prepare_training_data(normalized_data, target, weights)

        # Calculate batch size and gradient accumulation
        batch_size = data.n_samples

        # Prepare data loader with pre-calculated batch size
        dataloader = self._create_data_loader(
            x_tensor=self._train_data,
            y_tensor=self._train_target_classifier,
            weights_tensor=self._train_weights,
            batch_size=batch_size,
        )

        # Set up tensorboard logger
        tensorboard_log_dir = get_model_logging_dir(self._context, self._name)
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        # Use mixed precision (automatic fp16 casting) and optimized trainer settings
        trainer = pl.Trainer(
            max_epochs=self._config.train__epochs,
            logger=pl.loggers.TensorBoardLogger(
                save_dir=str(tensorboard_log_dir.parent),
                name=self._name,
            ),
            enable_progress_bar=True,
            enable_model_summary=True,
            precision='16-mixed' if torch.cuda.is_available() else '32',  # Auto mixed precision
            accumulate_grad_batches=1,  # Process each batch individually for stochastic noise
            num_sanity_val_steps=0,  # Skip validation sanity check
            log_every_n_steps=self._config.train__number_of_epochs_for_checkpoint,  # Logging frequency from config
        )

        # Training with binning context
        with self.binning_context(data):
            trainer.fit(self, dataloader)

        # Collect history from training
        return self._training_history

    def predict(self, data: DataSet) -> npt.NDArray:
        """
        Prediction method to be used with DataSet objects and one-time calculation of binning.
        ALREADY PERFORMING LOG! (until implemented in NPLM)
        """
        normalized_data = data / self._norm_factor
        with self.binning_context(data):
            x_tensor = torch.tensor(normalized_data.events, dtype=torch.float32, device=self.device)
            self.eval()
            with torch.no_grad():
                e_to_the_f_predictions = self(x_tensor, training=False)
            return torch.log(e_to_the_f_predictions).detach().cpu().numpy()

    def save_parameters(self, file_path) -> None:
        """Save PyTorch model parameters to file."""
        torch.save(self.state_dict(), file_path)


def calc_t_LFVNN(
        context: ExecutionContext,
        sample_dataset: DataSet,
        reference_dataset: DataSet,
        detector_effect: DetectorEffect,
        name: str,
) -> Tuple[pl.LightningModule, float]:
    
    feature = sample_dataset + reference_dataset
    target_classifier = np.concatenate((
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
        data=feature,
        target=target_classifier,
        weights=loss_weights,
    )
    
    info(f'Training time (seconds): {time() - t0}')
    
    # Calculate minimum loss from training history
    final_loss = tau_model_history[HistoryKeys.LOSS.value][-1]
    final_test_statistic = calc_t_test_statistic(final_loss)
    info(f'Minimum weighted loss achieved: {final_loss:.6f}')
    info(f'Observed t test statistic: {final_test_statistic}')
    
    save_training_outcomes(
        context,
        model_history=tau_model_history,
        tau_model=tau_model,
    )

    return tau_model, final_test_statistic
