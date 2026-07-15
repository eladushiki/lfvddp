from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from logging import info
from time import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import pad
from tqdm.auto import tqdm

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from data_tools.detector.detector_config import DetectorConfig
from data_tools.detector.detector_effect import DetectorEffect
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import (
    ContextedModel,
    save_model_parameters_outcome,
)
from train.checkpoints import find_latest_training_checkpoint, save_training_checkpoint
from train.train_config import TrainConfig


def _calculate_loss_weights(data: DataBatch) -> npt.NDArray:
    """
    Concatenate per-event detector weights in the canonical DataBatch order.

    Dataset-size coefficients belong to ``ddp_minimization_loss`` and must not
    be applied a second time here.
    """
    return np.concatenate(
        [dataset._weight_mask for dataset, _ in data],
        axis=0,
    )


class _ZeroEstimator(nn.Module):
    def __init__(self, output_dimension: int):
        super().__init__()
        self._output_dimension = output_dimension

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return events.new_zeros((events.shape[0], self._output_dimension))


class DifferentiatingModel(nn.Module, ContextedModel):
    """
    Symmetrized DDP's model used to estimate the test statistic using PyTorch Lightning.
    A custom loss function is used to find the maximizing parameters for hypothesis.
    """

    def __init__(
        self,
        context: ExecutionContext,
        detector_effect: DetectorEffect,
        is_numerator: bool,
        name: str,
    ):
        super().__init__()
        self._name = name
        self._is_numerator = is_numerator
        self._context = context
        self._config: Union[TrainConfig, DetectorConfig] = context.config

        # Add layers by spec. We would add two NNs to express f, g separately.
        self._build_layers()

        # Add detector uncertainty nuisance parameters
        self._detector_effect = detector_effect
        self._build_eta()
        self._bins_of_events = None  # Set in context

        # Initialize NN parameters according to strategy
        self._initialize_parameters()

        # Store training data and weights for metrics computation
        self._train_data = None
        self._train_target_classifier = None
        self._train_weights = None
        self._norm_factor = None
        self._training_history = defaultdict(list)
        self._a_sr_mask = None
        self._b_sr_mask = None
        self._a_cr_mask = None
        self._b_cr_mask = None

    @property
    def _device(self) -> torch.device:
        return torch.device("cpu")

    @property
    def _a_mask(self) -> torch.Tensor:
        if self._a_sr_mask is None or self._a_cr_mask is None:
            raise RuntimeError("Access to an uninitialized mask")
        return self._a_sr_mask | self._a_cr_mask

    @property
    def _b_mask(self) -> torch.Tensor:
        if self._b_sr_mask is None or self._b_cr_mask is None:
            raise RuntimeError("Access to an uninitialized mask")
        return self._b_sr_mask | self._b_cr_mask

    @property
    def _sr_mask(self) -> torch.Tensor:
        if self._a_sr_mask is None or self._b_sr_mask is None:
            raise RuntimeError("Access to an uninitialized mask")
        return self._a_sr_mask | self._b_sr_mask

    @property
    def _cr_mask(self) -> torch.Tensor:
        if self._a_cr_mask is None or self._b_cr_mask is None:
            raise RuntimeError("Access to an uninitialized mask")
        return self._a_cr_mask | self._b_cr_mask

    def _build_layers(self):
        # Fully connected 2-layer network:
        input_dim = self._config.train__nn_input_dimension
        hidden_size = self._config.train__nn_inner_layer_nodes
        output_size = self._config.train__nn_output_dimension

        if not self._is_numerator:
            self.f_network = _ZeroEstimator(output_size)
            self.g_network = _ZeroEstimator(output_size)
            return

        self.f_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )
        self.g_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
        )

    def _build_eta(self):
        self._detector_deltas = {}
        if not self._config.train__data_is_train_for_nuisances:
            self.eta = _ZeroEstimator(1)
            return

        for i, nbins in enumerate(self._detector_effect._numbers_of_bins):
            nuisance_var = nn.Parameter(
                torch.full((nbins,), 0.0, dtype=torch.float32, device=self._device)
            )  # Initialized later, value here has no meaning
            self.register_parameter(
                f"nuisance_{self._observable_names[i]}", nuisance_var
            )
            self._detector_deltas[self._observable_names[i]] = nuisance_var

        def eta(events: torch.Tensor) -> torch.Tensor:
            if self._bins_of_events is None:
                raise RuntimeError(
                    "eta can only be evaluated inside a binning_context."
                )
            dimensional_deltas = []
            for i, obs in enumerate(self._observable_names):
                dimensional_deltas.append(
                    self._detector_deltas[obs][self._bins_of_events[:, i]]
                )
            return torch.stack(dimensional_deltas, dim=1).prod(dim=1, keepdim=True)

        self.eta = eta

    def _initialize_parameters(self) -> None:
        """
        Create newly initialized weights matching the training strategy.
        This is the single source of truth for weight initialization.
        Assumes 2-layer network (1 hidden layer).
        """
        # Use Xavier uniform with configurable gain for weight initialization
        gain = self._config.train__nn_xavier_gain

        if self._is_numerator:
            for network in self.f_network, self.g_network:
                # Layer 0: Input to hidden
                hidden_layer = network[0]
                nn.init.xavier_uniform_(hidden_layer.weight, gain=gain)
                nn.init.uniform_(hidden_layer.bias, a=-0.3, b=0.3)

                # Layer 2: Hidden to output (skipping LeakyReLU at index 1)
                output_layer = network[2]
                nn.init.xavier_uniform_(output_layer.weight, gain=gain)
                nn.init.uniform_(output_layer.bias, a=-0.3, b=0.3)

        # Handle detector nuisances separately
        if self._config.train__data_is_train_for_nuisances:
            for var in self._detector_deltas.values():
                nn.init.normal_(var, mean=0.0, std=1e-3)

    def _clamp_nuisance_parameters(self) -> None:
        if not self._config.train__data_is_train_for_nuisances:
            return
        with torch.no_grad():
            for var in self._detector_deltas.values():
                var.clamp_(min=-1.0 + 1e-6, max=1.0 - 1e-6)

    def configure_optimizers(self) -> Optional[optim.Optimizer]:
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            return None
        optimizer = optim.Adam(
            trainable_parameters,
            lr=self._config.train__learning_rate,
        )
        return optimizer

    @property
    def _observable_names(self) -> List[str]:
        return self._detector_effect._observable_names

    def ddp_minimization_loss(
        self,
        f_of_x_sr_est: torch.Tensor,
        g_of_x_sr_est: torch.Tensor,
        eta_of_x_est: torch.Tensor,
    ) -> torch.Tensor:
        """
        The loss function for minimizing any expression in lfvddp.

        From the t values expression in the paper, use:
        - with or without f, g for numerator or denominator expression.
        - with or without nuisance parameters at will.
        Enter a zeros vector for each for it not to be use. This is the default
        behavior of DifferentiatingModel fit.

        Returns torch.Tensor of the same shape as either input to be summed.
        """
        # Setting the subsets we need for the element-wise loss calculation
        f_of_x_a_sr = f_of_x_sr_est * self._a_sr_mask
        g_of_x_b_sr = g_of_x_sr_est * self._b_sr_mask
        eta_a_sr = eta_of_x_est * self._a_sr_mask
        eta_b_sr = eta_of_x_est * self._b_sr_mask
        eta_sr = eta_a_sr + eta_b_sr
        eta_a_cr = eta_of_x_est * self._a_cr_mask
        eta_b_cr = eta_of_x_est * self._b_cr_mask
        eta_of_x_cr = eta_a_cr + eta_b_cr
        eta_of_x_a = eta_a_sr + eta_a_cr
        eta_of_x_b = eta_b_sr + eta_b_cr

        # Number constants
        N_A_SR = torch.count_nonzero(self._a_sr_mask)
        N_B_SR = torch.count_nonzero(self._b_sr_mask)
        N_A_CR = torch.count_nonzero(self._a_cr_mask)
        N_B_CR = torch.count_nonzero(self._b_cr_mask)
        N_SR = torch.count_nonzero(self._sr_mask)
        N_CR = torch.count_nonzero(self._cr_mask)

        e_to_the_f_sr = torch.exp(f_of_x_sr_est)
        eta_plus_term_sr = (1 + eta_sr) * self._sr_mask
        e_to_the_g_sr = torch.exp(g_of_x_sr_est)
        eta_minus_term_sr = (1 - eta_sr) * self._sr_mask
        sr_sum_term = (
            N_A_SR * e_to_the_f_sr * eta_plus_term_sr
            + N_B_SR * e_to_the_g_sr * eta_minus_term_sr
        ) / N_SR

        # CR sum term
        eta_plus_term_cr = (1 + eta_of_x_cr) * self._cr_mask
        eta_minus_term_cr = (1 - eta_of_x_cr) * self._cr_mask
        cr_sum_term = (N_A_CR * eta_plus_term_cr + N_B_CR * eta_minus_term_cr) / N_CR

        # eta log terms
        eta_plus_a_sum_term = torch.log(1 + eta_of_x_a)
        eta_minus_b_sum_term = torch.log(1 - eta_of_x_b)

        # Total loss is sum of log-likelihoods
        return (
            sr_sum_term
            + cr_sum_term
            - f_of_x_a_sr
            - g_of_x_b_sr
            - eta_plus_a_sum_term
            - eta_minus_b_sum_term
        )

    def forward(
        self,
        data: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        sr_data = data[self._sr_mask]
        f_of_x_sr_est = pad(
            self.f_network(sr_data).squeeze(),
            (0, data.numel() - sr_data.numel()),
        )
        g_of_x_sr_est = pad(
            self.g_network(sr_data).squeeze(),
            (0, data.numel() - sr_data.numel()),
        )
        eta_of_x_est = self.eta(data).squeeze()

        return torch.mean(
            (
                self.ddp_minimization_loss(
                    f_of_x_sr_est=f_of_x_sr_est,
                    g_of_x_sr_est=g_of_x_sr_est,
                    eta_of_x_est=eta_of_x_est,
                )
                * weights
            )
        )

    @contextmanager
    def binning_context(self, data: DataSet):
        try:
            self._bins_of_events = torch.tensor(
                self._detector_effect.get_event_bin_centers(data, indexed=True),
                dtype=torch.long,
                device=self._device,
            )
            yield
        finally:
            self._bins_of_events = None

    def _prepare_training_data(
        self,
        data: DataBatch,
        weights: npt.NDArray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Organize input data into masks and tensor for further processing.
        Initialize region masks for this train.
        """
        normalized_data, self._norm_factor = data.get_normalized()

        data_parts = []
        mask_parts = []
        for ds, params in normalized_data:
            data_parts.append(ds.events)
            mask_parts.append(
                np.full(ds.n_samples, params.category.value, dtype=np.int64)
            )

        data_tensor = torch.tensor(
            np.concatenate(data_parts), dtype=torch.float32, device=self._device
        )
        mask_tensor = torch.from_numpy(np.concatenate(mask_parts)).to(self._device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self._device)

        self._a_sr_mask = mask_tensor == DataSet.DataSetCategory.A_SR.value
        self._b_sr_mask = mask_tensor == DataSet.DataSetCategory.B_SR.value
        self._a_cr_mask = mask_tensor == DataSet.DataSetCategory.A_CR.value
        self._b_cr_mask = mask_tensor == DataSet.DataSetCategory.B_CR.value

        return data_tensor, weights_tensor

    def _log(self, epoch: int, loss: torch.Tensor) -> None:
        self._training_history[HistoryKeys.LOSS.value].append(
            float(loss.detach().cpu())
        )
        self._training_history[HistoryKeys.EPOCH.value].append(epoch)

    def _is_history_epoch(self, epoch: int) -> bool:
        return (
            (epoch + 1) % self._config.train__number_of_epochs_for_checkpoint == 0
            or epoch == self._config.train__epochs - 1
        )

    def _history_epochs(self) -> List[int]:
        return [
            epoch
            for epoch in range(self._config.train__epochs)
            if self._is_history_epoch(epoch)
        ]

    def has_trainable_parameters(self) -> bool:
        return any(parameter.requires_grad for parameter in self.parameters())

    def _load_training_checkpoint_if_requested(
        self, optimizer: Optional[optim.Optimizer]
    ) -> int:
        checkpoint_result = find_latest_training_checkpoint(
            self._context, self._name, warn_missing=False
        )
        if checkpoint_result is None:
            return 0

        checkpoint_path, checkpoint = checkpoint_result
        self.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state_dict = checkpoint.get("optimizer_state_dict")
        if optimizer is not None and optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        self._training_history = {
            key: list(value)
            for key, value in checkpoint.get("training_history", {}).items()
        }
        if (
            HistoryKeys.LOSS.value in self._training_history
            and HistoryKeys.EPOCH.value not in self._training_history
        ):
            self._training_history[HistoryKeys.EPOCH.value] = list(
                range(len(self._training_history[HistoryKeys.LOSS.value]))
            )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        info(
            f"Loaded checkpoint for {self._name} from {checkpoint_path}; resuming at epoch {start_epoch}"
        )
        return start_epoch

    def _train_step(
        self,
        optimizer: Optional[optim.Optimizer],
        data: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run one optimization step and return the batch loss."""
        loss = self(data=data, weights=weights)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self._clamp_nuisance_parameters()

        return loss

    def fit(
        self,
        data: DataBatch,
        weights: npt.NDArray,
    ) -> Dict[str, List[float]]:

        data_tensor, weights_tensor = self._prepare_training_data(data, weights)

        self.train()
        optimizer = self.configure_optimizers()

        target_epochs = self._config.train__epochs
        start_epoch = self._load_training_checkpoint_if_requested(optimizer)
        if start_epoch >= target_epochs:
            return self._training_history

        epoch_iterator = range(start_epoch, target_epochs)
        if self._config.train__enable_progress_bar:
            epoch_iterator = tqdm(epoch_iterator, desc=f"{self._name} training")

        # Training with binning context
        with self.binning_context(data.unified_data):
            for epoch in epoch_iterator:
                epoch_last_predictions = self._train_step(
                    optimizer=optimizer,
                    data=data_tensor,
                    weights=weights_tensor,
                )

                if self._is_history_epoch(epoch):
                    self._log(epoch, epoch_last_predictions)
                    save_training_checkpoint(
                        context=self._context,
                        model_name=self._name,
                        model=self,
                        optimizer=optimizer,
                        epoch=epoch,
                        training_history=self._training_history,
                    )

        # Collect history from training
        return self._training_history

    def calculate_loss_statically(
        self,
        data: DataBatch,
        weights: npt.NDArray,
    ) -> Dict[str, List[float]]:
        data_tensor, weights_tensor = self._prepare_training_data(data, weights)
        self.eval()
        with torch.no_grad():
            with self.binning_context(data.unified_data):
                loss = self(
                    data=data_tensor,
                    weights=weights_tensor,
                )

        epochs = self._history_epochs()
        loss_value = float(loss.detach().cpu())
        return {
            HistoryKeys.LOSS.value: [loss_value] * len(epochs),
            HistoryKeys.EPOCH.value: epochs,
        }

    def _predict_ndf(
        self,
        data: DataSet,
        network: nn.Module,
        eta_sign: float,
    ) -> npt.NDArray:
        if self._norm_factor is None:
            raise RuntimeError("Cannot predict before the model has been fitted.")
        normalized_data = data / self._norm_factor
        x_tensor = torch.tensor(
            normalized_data.events, dtype=torch.float32, device=self._device
        )
        self.eval()
        with torch.no_grad():
            with self.binning_context(data):
                eta_term = torch.clamp(1 + eta_sign * self.eta(x_tensor), min=1e-12)
                predictions = torch.exp(network(x_tensor)) * eta_term
        return predictions.detach().cpu().numpy()

    def predict(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, self.f_network, eta_sign=1.0)

    def predict_secondary(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, self.g_network, eta_sign=-1.0)

    def predict_eta(self, data: DataSet) -> npt.NDArray:
        x_tensor = torch.tensor(data.events, dtype=torch.float32, device=self._device)
        self.eval()
        with torch.no_grad():
            with self.binning_context(data):
                predictions = self.eta(x_tensor)
        return predictions.detach().cpu().numpy()

    def save_parameters(self, file_path) -> None:
        """Save PyTorch model parameters to file."""
        torch.save(self.state_dict(), file_path)


def calc_min_LFVNN(
    context: ExecutionContext,
    data: DataBatch,
    detector_effect: DetectorEffect,
    is_numerator: bool,
    name: str,
) -> Tuple[ContextedModel, float, Dict[str, List[float]]]:
    loss_weights = _calculate_loss_weights(data)

    model = DifferentiatingModel(
        context=context,
        detector_effect=detector_effect,
        is_numerator=is_numerator,
        name=name,
    )

    if not model.has_trainable_parameters():
        info("No trainable parameters in the model, calculating static expression.")
        model_history = model.calculate_loss_statically(
            data=data,
            weights=loss_weights,
        )

    else:
        info("Starting training")
        t0 = time()
        model_history = model.fit(
            data=data,
            weights=loss_weights,
        )
        info(f"Training time (seconds): {time() - t0}")

    # Calculate minimum loss from training history
    final_loss = model_history[HistoryKeys.LOSS.value][-1]
    info(f"Minimum weighted loss achieved: {final_loss:.6f}")

    save_model_parameters_outcome(context, model)

    return model, final_loss, model_history
