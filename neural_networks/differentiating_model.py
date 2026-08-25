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
from neural_networks.nuisance_calculation import (
    NeuralPerEventNuisanceCalculation, NoNuisanceCalculation, NuisanceEvaluation,
    PreparedNuisanceData, ScalarBinnedNuisanceCalculation, _ThetaEstimator,
)
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
    sr_data: torch.Tensor
    nuisance_data: PreparedNuisanceData
    number_of_a_sr_events: int
    number_of_b_sr_events: int
    number_of_a_cr_events: int
    number_of_b_cr_events: int
    a_sr_coefficient: float
    b_sr_coefficient: float
    cr_eta_coefficient: float
    @property
    def number_of_sr_events(self) -> int: return self.number_of_a_sr_events + self.number_of_b_sr_events
    @property
    def number_of_cr_events(self) -> int: return self.number_of_a_cr_events + self.number_of_b_cr_events


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


def _neural_nuisance_common_terms(
    eta_of_x_sr: torch.Tensor,
    eta_of_x_cr: torch.Tensor,
    number_of_a_sr_events: int,
    number_of_a_cr_events: int,
    number_of_cr_events: int,
    cr_eta_coefficient: float,
) -> torch.Tensor:
    """Evaluate per-event CR nuisance terms for neural nuisance parameters."""
    return (
        number_of_cr_events
        + cr_eta_coefficient * eta_of_x_cr.sum()
        - torch.log1p(eta_of_x_sr[:number_of_a_sr_events]).sum()
        - torch.log1p(eta_of_x_cr[:number_of_a_cr_events]).sum()
        - torch.log1p(-eta_of_x_sr[number_of_a_sr_events:]).sum()
        - torch.log1p(-eta_of_x_cr[number_of_a_cr_events:]).sum()
    )


def _neural_nuisance_loss(
    f_of_x_sr: torch.Tensor,
    g_of_x_sr: torch.Tensor,
    eta_of_x_sr: torch.Tensor,
    eta_of_x_cr: torch.Tensor,
    number_of_a_sr_events: int,
    number_of_a_cr_events: int,
    number_of_cr_events: int,
    a_sr_coefficient: float,
    b_sr_coefficient: float,
    cr_eta_coefficient: float,
) -> torch.Tensor:
    """Evaluate the numerator loss with per-event neural nuisance values."""
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
        + _neural_nuisance_common_terms(
            eta_of_x_sr=eta_of_x_sr,
            eta_of_x_cr=eta_of_x_cr,
            number_of_a_sr_events=number_of_a_sr_events,
            number_of_a_cr_events=number_of_a_cr_events,
            number_of_cr_events=number_of_cr_events,
            cr_eta_coefficient=cr_eta_coefficient,
        )
    )


def _neural_nuisance_denominator_loss(
    eta_of_x_sr: torch.Tensor,
    eta_of_x_cr: torch.Tensor,
    number_of_a_sr_events: int,
    number_of_a_cr_events: int,
    number_of_sr_events: int,
    number_of_cr_events: int,
    a_sr_coefficient: float,
    b_sr_coefficient: float,
    cr_eta_coefficient: float,
) -> torch.Tensor:
    """Evaluate the denominator loss with per-event neural nuisance values."""
    return (
        number_of_sr_events
        + (a_sr_coefficient - b_sr_coefficient) * eta_of_x_sr.sum()
        + _neural_nuisance_common_terms(
            eta_of_x_sr=eta_of_x_sr,
            eta_of_x_cr=eta_of_x_cr,
            number_of_a_sr_events=number_of_a_sr_events,
            number_of_a_cr_events=number_of_a_cr_events,
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
        self.nuisance_calculation = self._build_nuisance_calculation()

        # Add layers by spec. We would add two NNs to express f, g separately.
        self._build_layers()

        # Initialize NN parameters according to strategy
        self._initialize_parameters()
        self.to(self._assigned_device)

        self._norm_factor = None
        self._training_history = defaultdict(list)
        self._epochs_executed = 0

    @property
    def _device(self) -> torch.device:
        return self._assigned_device

    def _build_nuisance_calculation(self):
        if not self._config.train__data_is_train_for_nuisances:
            return NoNuisanceCalculation(self._dtype, self._device)
        if self._config.train__nuisance_is_neural_network:
            return NeuralPerEventNuisanceCalculation(self._config.train__nn_input_dimension, self._config.train__nuisance_nn_inner_layer_nodes, self._config.train__nn_output_dimension, self._dtype, self._device)
        return ScalarBinnedNuisanceCalculation(self._detector_effect, self._dtype, self._device)

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

        self.nuisance_calculation.initialize_parameters(gain)

    def _clamp_nuisance_parameters(self) -> None:
        self.nuisance_calculation.clamp_parameters()

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

    def _network_estimates(
        self,
        sr_data: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.paired_network is None:
            raise RuntimeError("The denominator has no f/g estimator.")
        f_estimate, g_estimate = self.paired_network(sr_data)
        return f_estimate.squeeze(-1), g_estimate.squeeze(-1)

    def forward(self, data: _PreparedTrainingData, profiler: Optional[TrainingProfiler] = None) -> torch.Tensor:
        profile_region = profiler.region if profiler is not None else nullcontext
        with profile_region("training/f_and_g_networks"):
            estimates = None if self.paired_network is None else self._network_estimates(data.sr_data)
        with profile_region("training/nuisance_eta"):
            nuisance = self.nuisance_calculation.evaluate(data.nuisance_data)
        return self._assemble_loss(estimates, nuisance, data)

    @staticmethod
    def _assemble_loss(
        estimates: Optional[tuple[torch.Tensor, torch.Tensor]],
        nuisance: NuisanceEvaluation,
        data: _PreparedTrainingData,
    ) -> torch.Tensor:
        common_terms = (
            data.number_of_cr_events
            + data.cr_eta_coefficient
            * torch.dot(
                nuisance.cr_values,
                nuisance.a_cr_weights + nuisance.b_cr_weights,
            )
            - torch.log1p(
                nuisance.sr_values[:data.number_of_a_sr_events]
            ).sum()
            - torch.dot(torch.log1p(nuisance.cr_values), nuisance.a_cr_weights)
            - torch.log1p(
                -nuisance.sr_values[data.number_of_a_sr_events:]
            ).sum()
            - torch.dot(torch.log1p(-nuisance.cr_values), nuisance.b_cr_weights)
        )
        if estimates is None:
            return (
                data.number_of_sr_events
                + (data.a_sr_coefficient - data.b_sr_coefficient)
                * nuisance.sr_values.sum()
                + common_terms
            )

        f_of_x_sr, g_of_x_sr = estimates
        a_term = data.a_sr_coefficient * torch.exp(f_of_x_sr)
        b_term = data.b_sr_coefficient * torch.exp(g_of_x_sr)
        return (
            torch.addcmul(
                a_term + b_term,
                nuisance.sr_values,
                a_term - b_term,
            ).sum()
            - f_of_x_sr[:data.number_of_a_sr_events].sum()
            - g_of_x_sr[data.number_of_a_sr_events:].sum()
            + common_terms
        )

    def _prepare_training_data(self, data: DataBatch) -> _PreparedTrainingData:
        normalized_data, self._norm_factor = data.get_normalized()
        categories = DataSet.DataSetCategory
        normalized_a_sr, normalized_b_sr = normalized_data.datasets[categories.A_SR], normalized_data.datasets[categories.B_SR]
        normalized_a_cr, normalized_b_cr = normalized_data.datasets[categories.A_CR], normalized_data.datasets[categories.B_CR]
        a_sr, b_sr, a_cr, b_cr = data.datasets[categories.A_SR], data.datasets[categories.B_SR], data.datasets[categories.A_CR], data.datasets[categories.B_CR]
        if a_sr.n_samples + b_sr.n_samples == 0: raise ValueError("Training requires at least one SR event.")
        if a_cr.n_samples + b_cr.n_samples == 0: raise ValueError("Training requires at least one CR event.")
        normalized_sr, raw_sr = DataSet(np.concatenate((normalized_a_sr.events, normalized_b_sr.events))), DataSet(np.concatenate((a_sr.events, b_sr.events)))
        sr_data = torch.tensor(normalized_sr.events, dtype=self._dtype, device=self._device)
        nuisance_data = self.nuisance_calculation.prepare(raw_sr, a_cr, b_cr, normalized_sr, normalized_a_cr, normalized_b_cr)
        number_of_sr_events, number_of_cr_events = a_sr.n_samples + b_sr.n_samples, a_cr.n_samples + b_cr.n_samples
        return _PreparedTrainingData(sr_data, nuisance_data, a_sr.n_samples, b_sr.n_samples, a_cr.n_samples, b_cr.n_samples, a_sr.n_samples / number_of_sr_events, b_sr.n_samples / number_of_sr_events, (a_cr.n_samples - b_cr.n_samples) / number_of_cr_events)

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

    def _set_learning_rate_for_epoch(
        self,
        optimizer: Optional[optim.Optimizer],
        epoch: int,
    ) -> None:
        """Set an adaptive learning rate for an absolute training epoch."""
        final_learning_rate = self._config.train__final_learning_rate
        if optimizer is None or final_learning_rate is None:
            return

        last_epoch = self._config.train__epochs - 1
        progress = epoch / last_epoch if last_epoch > 0 else 1.0
        learning_rate = (
            self._config.train__learning_rate
            + progress
            * (final_learning_rate - self._config.train__learning_rate)
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

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
                    self._set_learning_rate_for_epoch(optimizer, epoch)
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
                theta_inputs = (
                    x_tensor
                    if self.theta_network is not None
                    else self._bin_indices(data)
                )
                eta = self._theta_from_inputs(theta_inputs).unsqueeze(1)
            else:
                eta = x_tensor.new_zeros((x_tensor.shape[0], 1))
            eta_term = torch.clamp(1 + eta_sign * eta, min=1e-12)
            predictions = torch.exp(network_estimate) * eta_term
        return predictions.detach().cpu().numpy()

    def predict(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, secondary=False, eta_sign=1.0)

    def predict_secondary(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, secondary=True, eta_sign=-1.0)

    def predict_theta(self, data: DataSet) -> npt.NDArray:
        """Evaluate the configured nuisance function theta over a dataset."""
        self.eval()
        with torch.no_grad():
            if self._config.train__data_is_train_for_nuisances:
                if self.theta_network is not None:
                    if self._norm_factor is None:
                        raise RuntimeError("Cannot predict before the model has been fitted.")
                    normalized_data = data / self._norm_factor
                    theta_inputs = torch.tensor(
                        normalized_data.events, dtype=self._dtype, device=self._device
                    )
                else:
                    theta_inputs = self._bin_indices(data)
                predictions = self._theta_from_inputs(theta_inputs).unsqueeze(1)
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
