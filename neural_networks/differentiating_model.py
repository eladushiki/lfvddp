from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from logging import info
from time import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from train.training_profiler import TrainingProfiler


LFVNN_DTYPE = torch.float64


@dataclass(frozen=True)
class _PreparedTrainingData:
    """Static tensors and category sizes reused by every training epoch."""

    sr_data: torch.Tensor
    nuisance_bin_indices: Optional[torch.Tensor]
    a_cr_bin_counts: Optional[torch.Tensor]
    b_cr_bin_counts: Optional[torch.Tensor]
    number_of_a_sr_events: int
    number_of_b_sr_events: int
    number_of_a_cr_events: int
    number_of_b_cr_events: int
    a_sr_coefficient: float
    b_sr_coefficient: float
    cr_eta_coefficient: float

    @property
    def number_of_sr_events(self) -> int:
        return self.number_of_a_sr_events + self.number_of_b_sr_events

    @property
    def number_of_cr_events(self) -> int:
        return self.number_of_a_cr_events + self.number_of_b_cr_events

    @property
    def number_of_cr_bins(self) -> int:
        return 0 if self.a_cr_bin_counts is None else self.a_cr_bin_counts.numel()


class _PairedEstimator(nn.Module):
    """Evaluate the independent f and g networks with one shared input GEMM."""

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int,
        output_dimension: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if output_dimension != 1:
            raise ValueError("The paired estimator requires scalar f and g outputs.")
        self.hidden_size = hidden_size
        self.hidden = nn.Linear(input_dimension, 2 * hidden_size, dtype=dtype)
        self.activation = nn.Sigmoid()
        self.f_output = nn.Linear(hidden_size, output_dimension, dtype=dtype)
        self.g_output = nn.Linear(hidden_size, output_dimension, dtype=dtype)

    def forward(self, events: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.activation(self.hidden(events))
        paired_weight = torch.cat(
            (
                F.pad(self.f_output.weight, (0, self.hidden_size)),
                F.pad(self.g_output.weight, (self.hidden_size, 0)),
            ),
            dim=0,
        )
        paired_bias = torch.cat((self.f_output.bias, self.g_output.bias))
        estimates = F.linear(hidden, paired_weight, paired_bias)
        return estimates[:, :1], estimates[:, 1:]


def _compact_no_nuisance_loss(
    f_of_x_sr: torch.Tensor,
    g_of_x_sr: torch.Tensor,
    number_of_a_sr_events: int,
    number_of_cr_events: int,
    a_sr_coefficient: float,
    b_sr_coefficient: float,
) -> torch.Tensor:
    """Evaluate the unit-weight loss when detector nuisances are disabled."""
    return (
        a_sr_coefficient * torch.exp(f_of_x_sr).sum()
        + b_sr_coefficient * torch.exp(g_of_x_sr).sum()
        - f_of_x_sr[:number_of_a_sr_events].sum()
        - g_of_x_sr[number_of_a_sr_events:].sum()
        + number_of_cr_events
    )


def _compact_nuisance_common_terms(
    eta_of_x_sr: torch.Tensor,
    eta_of_x_cr_bins: torch.Tensor,
    a_cr_bin_counts: torch.Tensor,
    b_cr_bin_counts: torch.Tensor,
    number_of_a_sr_events: int,
    number_of_cr_events: int,
    cr_eta_coefficient: float,
) -> torch.Tensor:
    """Evaluate nuisance log terms and the compressed CR contribution."""
    eta_cr_sum = torch.dot(
        eta_of_x_cr_bins,
        a_cr_bin_counts + b_cr_bin_counts,
    )
    return (
        number_of_cr_events
        + cr_eta_coefficient * eta_cr_sum
        - torch.log1p(eta_of_x_sr[:number_of_a_sr_events]).sum()
        - torch.dot(torch.log1p(eta_of_x_cr_bins), a_cr_bin_counts)
        - torch.log1p(-eta_of_x_sr[number_of_a_sr_events:]).sum()
        - torch.dot(torch.log1p(-eta_of_x_cr_bins), b_cr_bin_counts)
    )


def _compact_nuisance_loss(
    f_of_x_sr: torch.Tensor,
    g_of_x_sr: torch.Tensor,
    eta_of_x_sr: torch.Tensor,
    eta_of_x_cr_bins: torch.Tensor,
    a_cr_bin_counts: torch.Tensor,
    b_cr_bin_counts: torch.Tensor,
    number_of_a_sr_events: int,
    number_of_cr_events: int,
    a_sr_coefficient: float,
    b_sr_coefficient: float,
    cr_eta_coefficient: float,
) -> torch.Tensor:
    """Evaluate the unit-weight numerator loss with compressed CR bins."""
    a_sr_term = a_sr_coefficient * torch.exp(f_of_x_sr)
    b_sr_term = b_sr_coefficient * torch.exp(g_of_x_sr)
    sr_density = torch.addcmul(
        a_sr_term + b_sr_term,
        eta_of_x_sr,
        a_sr_term - b_sr_term,
    )

    return (
        sr_density.sum()
        - f_of_x_sr[:number_of_a_sr_events].sum()
        - g_of_x_sr[number_of_a_sr_events:].sum()
        + _compact_nuisance_common_terms(
            eta_of_x_sr=eta_of_x_sr,
            eta_of_x_cr_bins=eta_of_x_cr_bins,
            a_cr_bin_counts=a_cr_bin_counts,
            b_cr_bin_counts=b_cr_bin_counts,
            number_of_a_sr_events=number_of_a_sr_events,
            number_of_cr_events=number_of_cr_events,
            cr_eta_coefficient=cr_eta_coefficient,
        )
    )


def _compact_nuisance_denominator_loss(
    eta_of_x_sr: torch.Tensor,
    eta_of_x_cr_bins: torch.Tensor,
    a_cr_bin_counts: torch.Tensor,
    b_cr_bin_counts: torch.Tensor,
    number_of_a_sr_events: int,
    number_of_sr_events: int,
    number_of_cr_events: int,
    a_sr_coefficient: float,
    b_sr_coefficient: float,
    cr_eta_coefficient: float,
) -> torch.Tensor:
    """Evaluate the nuisance denominator without zero-network tensors."""
    return (
        number_of_sr_events
        + (a_sr_coefficient - b_sr_coefficient) * eta_of_x_sr.sum()
        + _compact_nuisance_common_terms(
            eta_of_x_sr=eta_of_x_sr,
            eta_of_x_cr_bins=eta_of_x_cr_bins,
            a_cr_bin_counts=a_cr_bin_counts,
            b_cr_bin_counts=b_cr_bin_counts,
            number_of_a_sr_events=number_of_a_sr_events,
            number_of_cr_events=number_of_cr_events,
            cr_eta_coefficient=cr_eta_coefficient,
        )
    )


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
        dtype: torch.dtype = LFVNN_DTYPE,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self._context = context
        self._config: Union[TrainConfig, DetectorConfig] = context.config
        self._detector_effect = detector_effect
        self._is_numerator = is_numerator
        self._name = name
        self._dtype = dtype
        self._assigned_device = torch.device(device)

        # Add layers by spec. We would add two NNs to express f, g separately.
        self._build_layers()

        # Add detector uncertainty nuisance parameters
        self._build_eta()

        # Initialize NN parameters according to strategy
        self._initialize_parameters()
        self.to(self._assigned_device)

        self._norm_factor = None
        self._training_history = defaultdict(list)
        self._epochs_executed = 0

    @property
    def _device(self) -> torch.device:
        return self._assigned_device

    def _build_layers(self):
        self.paired_network = (
            _PairedEstimator(
                input_dimension=self._config.train__nn_input_dimension,
                hidden_size=self._config.train__nn_inner_layer_nodes,
                output_dimension=self._config.train__nn_output_dimension,
                dtype=self._dtype,
            )
            if self._is_numerator
            else None
        )

    def _build_eta(self):
        self._detector_deltas = {}
        if not self._config.train__data_is_train_for_nuisances:
            return

        for i, nbins in enumerate(self._detector_effect._numbers_of_bins):
            nuisance_var = nn.Parameter(
                torch.full(
                    (nbins,),
                    0.0,
                    dtype=self._dtype,
                    device=self._device,
                )
            )  # Initialized later, value here has no meaning
            self.register_parameter(
                f"nuisance_{self._observable_names[i]}", nuisance_var
            )
            self._detector_deltas[self._observable_names[i]] = nuisance_var

    def _initialize_parameters(self) -> None:
        """
        Create newly initialized weights matching the training strategy.
        This is the single source of truth for weight initialization.
        Assumes 2-layer network (1 hidden layer).
        """
        # Use Xavier uniform with configurable gain for weight initialization
        gain = self._config.train__nn_xavier_gain

        if self.paired_network is not None:
            hidden_size = self.paired_network.hidden_size
            for hidden_slice, output_layer in (
                (
                    slice(0, hidden_size),
                    self.paired_network.f_output,
                ),
                (
                    slice(hidden_size, 2 * hidden_size),
                    self.paired_network.g_output,
                ),
            ):
                nn.init.xavier_uniform_(
                    self.paired_network.hidden.weight[hidden_slice],
                    gain=gain,
                )
                nn.init.uniform_(
                    self.paired_network.hidden.bias[hidden_slice],
                    a=-0.3,
                    b=0.3,
                )
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

    def _bin_indices(self, data: DataSet) -> torch.Tensor:
        return torch.tensor(
            self._detector_effect.get_event_bin_centers(data, indexed=True),
            dtype=torch.long,
            device=self._device,
        )

    def _eta_from_bin_indices(self, bin_indices: torch.Tensor) -> torch.Tensor:
        if not self._config.train__data_is_train_for_nuisances:
            return torch.zeros(
                bin_indices.shape[0],
                dtype=self._dtype,
                device=self._device,
            )

        eta: Optional[torch.Tensor] = None
        for dimension, observable_name in enumerate(self._observable_names):
            dimensional_eta = torch.index_select(
                self._detector_deltas[observable_name],
                dim=0,
                index=bin_indices[:, dimension],
            )
            eta = dimensional_eta if eta is None else eta * dimensional_eta
        if eta is None:
            raise RuntimeError("At least one detector observable is required.")
        return eta

    def _compressed_cr_bin_indices(
        self,
        a_cr: DataSet,
        b_cr: DataSet,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return shared unique CR bins and per-category event counts."""
        cr_bin_indices = torch.cat(
            (self._bin_indices(a_cr), self._bin_indices(b_cr)),
            dim=0,
        )
        unique_bin_indices, inverse_indices = torch.unique(
            cr_bin_indices,
            dim=0,
            return_inverse=True,
        )
        number_of_unique_bins = unique_bin_indices.shape[0]
        a_cr_bin_counts = torch.bincount(
            inverse_indices[: a_cr.n_samples],
            minlength=number_of_unique_bins,
        ).to(dtype=self._dtype)
        b_cr_bin_counts = torch.bincount(
            inverse_indices[a_cr.n_samples :],
            minlength=number_of_unique_bins,
        ).to(dtype=self._dtype)
        return unique_bin_indices, a_cr_bin_counts, b_cr_bin_counts

    def _network_estimates(
        self,
        sr_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.paired_network is None:
            raise RuntimeError("The denominator has no f/g estimator.")
        f_estimate, g_estimate = self.paired_network(sr_data)
        return f_estimate.squeeze(-1), g_estimate.squeeze(-1)

    def forward(
        self,
        data: _PreparedTrainingData,
        profiler: Optional[TrainingProfiler] = None,
    ) -> torch.Tensor:
        profile_region = profiler.region if profiler is not None else nullcontext
        with profile_region("training/f_and_g_networks"):
            if self.paired_network is None:
                f_of_x_sr_est = None
                g_of_x_sr_est = None
            else:
                f_of_x_sr_est, g_of_x_sr_est = self._network_estimates(
                    data.sr_data
                )

        with profile_region("training/nuisance_eta"):
            if self._config.train__data_is_train_for_nuisances:
                if data.nuisance_bin_indices is None:
                    raise RuntimeError("Training bin indices were not prepared.")
                eta_values = self._eta_from_bin_indices(
                    data.nuisance_bin_indices
                )
                eta_of_x_sr, eta_of_x_cr_bins = torch.split(
                    eta_values,
                    (
                        data.number_of_sr_events,
                        data.number_of_cr_bins,
                    ),
                )
            else:
                eta_of_x_sr = None
                eta_of_x_cr_bins = None

        with profile_region("training/loss_expression"):
            if eta_of_x_sr is None:
                if f_of_x_sr_est is None or g_of_x_sr_est is None:
                    return data.sr_data.new_tensor(
                        data.number_of_sr_events + data.number_of_cr_events
                    )
                return _compact_no_nuisance_loss(
                    f_of_x_sr=f_of_x_sr_est,
                    g_of_x_sr=g_of_x_sr_est,
                    number_of_a_sr_events=data.number_of_a_sr_events,
                    number_of_cr_events=data.number_of_cr_events,
                    a_sr_coefficient=data.a_sr_coefficient,
                    b_sr_coefficient=data.b_sr_coefficient,
                )
            if (
                eta_of_x_cr_bins is None
                or data.a_cr_bin_counts is None
                or data.b_cr_bin_counts is None
            ):
                raise RuntimeError("Compressed CR nuisance data was not prepared.")
            if f_of_x_sr_est is None or g_of_x_sr_est is None:
                return _compact_nuisance_denominator_loss(
                    eta_of_x_sr=eta_of_x_sr,
                    eta_of_x_cr_bins=eta_of_x_cr_bins,
                    a_cr_bin_counts=data.a_cr_bin_counts,
                    b_cr_bin_counts=data.b_cr_bin_counts,
                    number_of_a_sr_events=data.number_of_a_sr_events,
                    number_of_sr_events=data.number_of_sr_events,
                    number_of_cr_events=data.number_of_cr_events,
                    a_sr_coefficient=data.a_sr_coefficient,
                    b_sr_coefficient=data.b_sr_coefficient,
                    cr_eta_coefficient=data.cr_eta_coefficient,
                )
            return _compact_nuisance_loss(
                f_of_x_sr=f_of_x_sr_est,
                g_of_x_sr=g_of_x_sr_est,
                eta_of_x_sr=eta_of_x_sr,
                eta_of_x_cr_bins=eta_of_x_cr_bins,
                a_cr_bin_counts=data.a_cr_bin_counts,
                b_cr_bin_counts=data.b_cr_bin_counts,
                number_of_a_sr_events=data.number_of_a_sr_events,
                number_of_cr_events=data.number_of_cr_events,
                a_sr_coefficient=data.a_sr_coefficient,
                b_sr_coefficient=data.b_sr_coefficient,
                cr_eta_coefficient=data.cr_eta_coefficient,
            )

    def _prepare_training_data(
        self,
        data: DataBatch,
    ) -> _PreparedTrainingData:
        """Prepare compact, immutable tensors for repeated full-batch training."""
        normalized_data, self._norm_factor = data.get_normalized()
        categories = DataSet.DataSetCategory
        normalized_a_sr = normalized_data.datasets[categories.A_SR]
        normalized_b_sr = normalized_data.datasets[categories.B_SR]
        a_sr = data.datasets[categories.A_SR]
        b_sr = data.datasets[categories.B_SR]
        a_cr = data.datasets[categories.A_CR]
        b_cr = data.datasets[categories.B_CR]

        if a_sr.n_samples + b_sr.n_samples == 0:
            raise ValueError("Training requires at least one SR event.")
        if a_cr.n_samples + b_cr.n_samples == 0:
            raise ValueError("Training requires at least one CR event.")

        sr_data = torch.tensor(
            np.concatenate((normalized_a_sr.events, normalized_b_sr.events)),
            dtype=self._dtype,
            device=self._device,
        )
        if self._config.train__data_is_train_for_nuisances:
            sr_bin_indices = torch.cat(
                (self._bin_indices(a_sr), self._bin_indices(b_sr)),
                dim=0,
            )
            (
                cr_bin_indices,
                a_cr_bin_counts,
                b_cr_bin_counts,
            ) = self._compressed_cr_bin_indices(a_cr, b_cr)
            nuisance_bin_indices = torch.cat(
                (sr_bin_indices, cr_bin_indices),
                dim=0,
            )
        else:
            nuisance_bin_indices = None
            a_cr_bin_counts = None
            b_cr_bin_counts = None

        number_of_sr_events = a_sr.n_samples + b_sr.n_samples
        number_of_cr_events = a_cr.n_samples + b_cr.n_samples

        return _PreparedTrainingData(
            sr_data=sr_data,
            nuisance_bin_indices=nuisance_bin_indices,
            a_cr_bin_counts=a_cr_bin_counts,
            b_cr_bin_counts=b_cr_bin_counts,
            number_of_a_sr_events=a_sr.n_samples,
            number_of_b_sr_events=b_sr.n_samples,
            number_of_a_cr_events=a_cr.n_samples,
            number_of_b_cr_events=b_cr.n_samples,
            a_sr_coefficient=a_sr.n_samples / number_of_sr_events,
            b_sr_coefficient=b_sr.n_samples / number_of_sr_events,
            cr_eta_coefficient=(a_cr.n_samples - b_cr.n_samples)
            / number_of_cr_events,
        )

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
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(self._device)
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
        data: _PreparedTrainingData,
        profiler: TrainingProfiler,
    ) -> torch.Tensor:
        """Run one optimization step and return the batch loss."""
        with profiler.region("training/forward_and_loss"):
            loss = self(data=data, profiler=profiler)

        if optimizer is not None:
            with profiler.region("training/zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            with profiler.region("training/backward"):
                loss.backward()
            with profiler.region("training/optimizer_step"):
                optimizer.step()
            with profiler.region("training/clamp_nuisances"):
                self._clamp_nuisance_parameters()

        return loss

    def fit(
        self,
        data: DataBatch,
    ) -> Dict[str, List[float]]:
        training_data = self._prepare_training_data(data)

        self.train()
        optimizer = self.configure_optimizers()

        target_epochs = self._config.train__epochs
        start_epoch = self._load_training_checkpoint_if_requested(optimizer)
        if start_epoch >= target_epochs:
            return self._training_history
        self._epochs_executed = target_epochs - start_epoch

        epoch_iterator = range(start_epoch, target_epochs)
        if self._config.train__enable_progress_bar:
            epoch_iterator = tqdm(epoch_iterator, desc=f"{self._name} training")

        profiler = TrainingProfiler(
            context=self._context,
            model_name=self._name,
            number_of_observables=data.unified_data.n_observables,
            number_of_events=data.unified_data.n_samples,
            number_of_training_epochs=target_epochs - start_epoch,
            device=self._device,
        )
        with profiler:
            for epoch in epoch_iterator:
                with profiler.region("training/epoch"):
                    epoch_last_predictions = self._train_step(
                        optimizer=optimizer,
                        data=training_data,
                        profiler=profiler,
                    )

                    if self._is_history_epoch(epoch):
                        with profiler.region("training/history"):
                            self._log(epoch, epoch_last_predictions)
                        with profiler.region("training/checkpoint"):
                            save_training_checkpoint(
                                context=self._context,
                                model_name=self._name,
                                model=self,
                                optimizer=optimizer,
                                epoch=epoch,
                                training_history=self._training_history,
                            )
                profiler.step()

        # Collect history from training
        return self._training_history

    def calculate_loss_statically(
        self,
        data: DataBatch,
    ) -> Dict[str, List[float]]:
        training_data = self._prepare_training_data(data)
        self.eval()
        with torch.no_grad():
            loss = self(data=training_data)

        epochs = self._history_epochs()
        loss_value = float(loss.detach().cpu())
        return {
            HistoryKeys.LOSS.value: [loss_value] * len(epochs),
            HistoryKeys.EPOCH.value: epochs,
        }

    def _predict_ndf(
        self,
        data: DataSet,
        secondary: bool,
        eta_sign: float,
    ) -> npt.NDArray:
        if self._norm_factor is None:
            raise RuntimeError("Cannot predict before the model has been fitted.")
        normalized_data = data / self._norm_factor
        x_tensor = torch.tensor(
            normalized_data.events,
            dtype=self._dtype,
            device=self._device,
        )
        self.eval()
        with torch.no_grad():
            if self.paired_network is None:
                network_estimate = x_tensor.new_zeros((x_tensor.shape[0], 1))
            else:
                f_estimate, g_estimate = self.paired_network(x_tensor)
                network_estimate = g_estimate if secondary else f_estimate
            if self._config.train__data_is_train_for_nuisances:
                eta = self._eta_from_bin_indices(
                    self._bin_indices(data)
                ).unsqueeze(1)
            else:
                eta = x_tensor.new_zeros((x_tensor.shape[0], 1))
            eta_term = torch.clamp(1 + eta_sign * eta, min=1e-12)
            predictions = torch.exp(network_estimate) * eta_term
        return predictions.detach().cpu().numpy()

    def predict(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, secondary=False, eta_sign=1.0)

    def predict_secondary(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, secondary=True, eta_sign=-1.0)

    def predict_eta(self, data: DataSet) -> npt.NDArray:
        self.eval()
        with torch.no_grad():
            if self._config.train__data_is_train_for_nuisances:
                predictions = self._eta_from_bin_indices(
                    self._bin_indices(data)
                ).unsqueeze(1)
            else:
                predictions = torch.zeros(
                    (data.n_samples, 1), dtype=self._dtype, device=self._device
                )
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
    device: Union[str, torch.device] = "cpu",
) -> Tuple[ContextedModel, float, Dict[str, List[float]]]:
    model = DifferentiatingModel(
        context=context,
        detector_effect=detector_effect,
        is_numerator=is_numerator,
        name=name,
        device=device,
    )

    if not model.has_trainable_parameters():
        info("No trainable parameters in the model, calculating static expression.")
        model_history = model.calculate_loss_statically(data=data)

    else:
        info("Starting training")
        t0 = time()
        model_history = model.fit(data=data)
        if model._device.type == "cuda":
            torch.cuda.synchronize(model._device)
        info(f"Training time (seconds): {time() - t0}")

    # Calculate minimum loss from training history
    final_loss = model_history[HistoryKeys.LOSS.value][-1]
    info(f"Minimum loss achieved: {final_loss:.6f}")

    save_model_parameters_outcome(context, model)

    return model, final_loss, model_history
