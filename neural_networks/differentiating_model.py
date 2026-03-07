from __future__ import annotations

from contextlib import contextmanager
from logging import info
from time import time
from typing import Dict, List, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from data_tools.detector.detector_effect import DetectorEffect
from data_tools.data_utils import DataSet
from data_tools.detector.constants import TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import ContextedModel, save_training_outcomes, get_model_logging_dir
from train.train_config import TrainConfig


class DifferentiatingModel(nn.Module, ContextedModel):
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
        self._tensorboard_writer = None

    @property
    def _device(self) -> torch.device:
        return torch.device("cpu")

    def _build_layers(self):
        # Fully connected 2-layer network:
        input_dim = self._config.train__nn_input_dimension
        hidden_size = self._config.train__nn_inner_layer_nodes
        output_size = self._config.train__nn_output_dimension
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )

    def _initialize_tensorboard_writer(self) -> None:
        """Initialize TensorBoard writer for this model instance."""
        tensorboard_log_dir = get_model_logging_dir(self._context, self._name)
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self._tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_log_dir))

    def _close_tensorboard_writer(self) -> None:
        """Close and clear TensorBoard writer if initialized."""
        if self._tensorboard_writer is not None:
            self._tensorboard_writer.close()
            self._tensorboard_writer = None

    def _build_detector_nuisances(self):
        self._detector_deltas = {}
        for i, nbins in enumerate(self._detector_effect._numbers_of_bins):
            if self._config.train__data_is_train_for_nuisances:
                nuisance_var = nn.Parameter(
                    torch.zeros(nbins, dtype=torch.float32, device=self._device)
                )
            else:
                nuisance_var = torch.zeros(nbins, dtype=torch.float32, device=self._device)
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
        nn.init.uniform_(hidden_layer.bias, a=-0.3, b=0.3)
        
        # Layer 2: Hidden to output (skipping LeakyReLU at index 1)
        output_layer = self.network[2]
        nn.init.xavier_uniform_(output_layer.weight, gain=gain)
        nn.init.uniform_(output_layer.bias, a=-0.3, b=0.3)

        # Handle detector nuisances separately
        if self._config.train__data_is_train_for_nuisances:
            for var in self._detector_deltas.values():
                nn.init.normal_(var, mean=0.0, std=float(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD))

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self._config.train__learning_rate,
        )

        return optimizer

    def _gaussian_nuisance_nll(self, nuisance_value: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood of a single (or vector of) nuisance parameter(s) under
        Gaussian constraint.
        - log(x) = 0.5 * (x/σ)² + log(σ√(2π))
        Constant term is dropped.
        return: torch.Tensor: Tensor of same shape as nuisance_value with NLL values.
        """
        std = torch.tensor(TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD, dtype=torch.float32, device=self._device)
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
            f_prediction: torch.Tensor,
        ) -> torch.Tensor:
        """
        The custom negative log-likelihood for the prediction of the NN.
        Rewards correct classification of sample vs. reference events.
        return: torch.Tensor: Tensor of same shape as input tensors with NLL values.
        """
        is_ref_classifier = 1.0 - is_sample_classifier
        return is_ref_classifier * (torch.exp(f_prediction) - 1) \
            - is_sample_classifier * f_prediction

    @property
    def _observable_names(self) -> List[str]:
        return self._detector_effect._observable_names

    def ddp_symmetrized_loss(
            self,
            is_sample_classifier: torch.Tensor,
            f_prediction: torch.Tensor,
        ) -> torch.Tensor:
        """
        Symmetrized DDP custom loss for optimizing likelihood of the
        estimation. Returns negative log-likelihood to be minimized.
        return: torch.Tensor: Tensor of same shape as input tensors with total NLL values.
        """
        prediction_loss = self._prediction_nll(
            is_sample_classifier=is_sample_classifier,
            f_prediction=f_prediction,
        )  # Tensor the size of data
        if self._config.train__data_is_train_for_nuisances:
            nuisance_loss = self._total_nuisance_nll()  # Scalar
        else:
            nuisance_loss = torch.tensor(0.0, device=self._device, dtype=torch.float32)

        # Total loss is sum of log-likelihoods
        return prediction_loss + nuisance_loss

    def forward(self, data: torch.Tensor, training: bool = True) -> torch.Tensor:
        f_prdiction = self.network(data)

        # Each event predicted weight is multiplied by the exponentiation multiplication of all affecting nuisances
        if self._config.train__data_is_train_for_nuisances:
            nuisance_skews = [
                torch.gather(self._detector_deltas[obs], 0, self._bins_of_events[:, i])
                for i, obs in enumerate(self._observable_names)
            ]

            items = torch.stack([f_prdiction.squeeze(), *nuisance_skews])
            return torch.prod(items, dim=0)
        else:
            return f_prdiction.squeeze()

    @contextmanager
    def binning_context(self, data: DataSet):
        try:
            self._bins_of_events = torch.tensor(
                self._detector_effect.get_event_bin_centers(data, indexed=True),
                dtype=torch.long,
                device=self._device
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
        data_tensor = torch.tensor(data.events, dtype=torch.float32, device=self._device)
        target_classifier_tensor = torch.tensor(target_classifier, dtype=torch.float32, device=self._device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self._device)
        return data_tensor, target_classifier_tensor, weights_tensor

    def _should_log_epoch(self, epoch: int, max_epochs: int) -> bool:
        checkpoint_interval = self._config.train__number_of_epochs_for_checkpoint
        is_checkpoint_epoch = (epoch + 1) % checkpoint_interval == 0
        is_last_epoch = (epoch + 1) >= max_epochs
        return is_checkpoint_epoch or is_last_epoch

    def _calculate_metrics(
        self,
        is_sample_classifier: torch.Tensor,
        f_prediction: torch.Tensor,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Calculate all metrics in a structured manner.
        Returns a dict with all relevant metric values.
        """
        metrics = {}
        metrics[HistoryKeys.EPOCH.value] = epoch

        # Prediction loss
        prediction_nll = self._prediction_nll(
            is_sample_classifier,
            f_prediction,
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

    def _log_epoch_metrics(self, epoch: int, f_prediction: torch.Tensor | None = None) -> None:
        """Compute metrics and persist scalar/histogram logs for a given epoch."""
        if f_prediction is None:
            with torch.no_grad():
                f_prediction = self(self._train_data, training=False)

        logs = self._calculate_metrics(
            is_sample_classifier=self._train_target_classifier,
            f_prediction=f_prediction,
            epoch=epoch,
        )

        for metric_name, metric_value in logs.items():
            self._tensorboard_writer.add_scalar(metric_name, metric_value, epoch)
            if metric_name not in self._training_history:
                self._training_history[metric_name] = []
            self._training_history[metric_name].append(metric_value)

        if self._context.is_debug_mode:
            for name, param in self.network.named_parameters():
                self._tensorboard_writer.add_histogram(f'weights/{name}', param, epoch)

    def _train_step(
        self,
        optimizer: optim.Optimizer,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        batch_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run one optimization step and return detached predictions for this batch."""
        batch_f_predictions = self(batch_x, training=True)
        per_sample_loss = self.ddp_symmetrized_loss(
            is_sample_classifier=batch_y,
            f_prediction=batch_f_predictions,
        )
        weighted_loss = per_sample_loss * batch_weights
        loss = torch.mean(weighted_loss)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return self(batch_x, training=False).detach()

    def fit(
        self,
        data: DataSet,
        target: npt.NDArray,
        weights: npt.NDArray,
    ) -> Dict[str, List[float]]:
        """Pure PyTorch training loop."""
        # Store training data for metrics computation
        normalized_data, self._norm_factor = data.get_normalized()
        self._train_data, self._train_target_classifier, self._train_weights = self._prepare_training_data(normalized_data, target, weights)

        # Full-batch mode only: use direct tensors to avoid DataLoader/collate overhead.
        batch_x = self._train_data
        batch_y = self._train_target_classifier
        batch_weights = self._train_weights

        self.train()
        self._initialize_tensorboard_writer()
        optimizer = self.configure_optimizers()

        max_epochs = self._config.train__epochs
        epoch_iterator = range(max_epochs)
        if self._config.train__enable_progress_bar:
            epoch_iterator = tqdm(epoch_iterator, desc=f"{self._name} training")

        # Training with binning context
        with self.binning_context(data):
            for epoch in epoch_iterator:
                epoch_last_predictions = self._train_step(
                    optimizer=optimizer,
                    batch_x=batch_x,
                    batch_y=batch_y,
                    batch_weights=batch_weights,
                )

                if self._should_log_epoch(epoch=epoch, max_epochs=max_epochs):
                    self._log_epoch_metrics(
                        epoch=epoch,
                        f_prediction=epoch_last_predictions,
                    )

        self._close_tensorboard_writer()

        # Collect history from training
        return self._training_history

    def predict(self, data: DataSet) -> npt.NDArray:
        """
        Prediction method to be used with DataSet objects and one-time calculation of binning.
        ALREADY PERFORMING LOG! (until implemented in NPLM)
        """
        normalized_data = data / self._norm_factor
        with self.binning_context(data):
            x_tensor = torch.tensor(normalized_data.events, dtype=torch.float32, device=self._device)
            self.eval()
            with torch.no_grad():
                f_predictions = self(x_tensor, training=False)
            return f_predictions.detach().cpu().numpy()

    def save_parameters(self, file_path) -> None:
        """Save PyTorch model parameters to file."""
        torch.save(self.state_dict(), file_path)


def calc_t_LFVNN(
        context: ExecutionContext,
        sample_dataset: DataSet,
        reference_dataset: DataSet,
        detector_effect: DetectorEffect,
        name: str,
) -> Tuple[ContextedModel, float]:
    
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
