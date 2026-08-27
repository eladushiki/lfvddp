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
from torch import nn, optim
from tqdm.auto import tqdm

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from data_tools.detector.detector_config import DetectorConfig
from data_tools.detector.detector_effect import DetectorEffect
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.nuisance_calculation import (
    BlankNuisanceEstimator,
    NeuralPerEventNuisanceEstimator,
    NuisanceEvaluation,
    PreparedNuisanceData,
    ScalarBinnedNuisanceEstimator,
)
from neural_networks.utils import (
    ContextedModel,
    save_model_parameters_outcome,
)
from train.checkpoints import find_latest_training_checkpoint, save_training_checkpoint
from train.train_config import TrainConfig
from train.training_profiler import TrainingProfiler

LFVNN_DTYPE = torch.float64
_SIGNAL_REGION_SHIFT_BOUND = 1.0 - 1e-6


@dataclass(frozen=True)
class _PreparedTrainingData:
    sr_events: torch.Tensor
    nuisance_data: PreparedNuisanceData
    a_sr_mask: torch.Tensor
    b_sr_mask: torch.Tensor
    N_a_sr: int
    N_b_sr: int
    N_a_cr: int
    N_b_cr: int
    n_a_sr_over_n_sr: float
    n_b_sr_over_n_sr: float
    nuisance_cr_coefficient: float

    @property
    def N_sr(self) -> int:
        return self.N_a_sr + self.N_b_sr

    @property
    def number_of_cr_events(self) -> int:
        return self.N_a_cr + self.N_b_cr


class _SignalRegionShiftEstimator(nn.Module):
    """Estimate the single bounded signal-region shift f for both categories."""

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int,
        output_dimension: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_dimension, hidden_size, dtype=dtype)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(hidden_size, output_dimension, dtype=dtype)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(events))).clamp(
            min=-_SIGNAL_REGION_SHIFT_BOUND,
            max=_SIGNAL_REGION_SHIFT_BOUND,
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
        self.nuisance_calculation = self._build_nuisance_estimators()

        self._build_signal_hypothesis_estimator()

        # Initialize NN parameters according to strategy
        self._initialize_parameters()
        self.to(self._assigned_device)

        self._norm_factor = None
        self._training_history = defaultdict(list)
        self._epochs_executed = 0

    @property
    def _device(self) -> torch.device:
        return self._assigned_device

    def _build_nuisance_estimators(self):
        if not self._config.train__data_is_train_for_nuisances:
            return BlankNuisanceEstimator(
                dtype=self._dtype,
                device=self._device,
            )
        if self._config.train__nuisance_is_neural_network:
            return NeuralPerEventNuisanceEstimator(
                input_dimension=self._config.train__nn_input_dimension,
                hidden_size=self._config.train__nuisance_nn_inner_layer_nodes,
                output_dimension=self._config.train__nn_output_dimension,
                dtype=self._dtype,
                device=self._device,
            )
        return ScalarBinnedNuisanceEstimator(
            detector_effect=self._detector_effect,
            dtype=self._dtype,
            device=self._device,
        )

    def _build_signal_hypothesis_estimator(self) -> None:
        self.signal_region_shift_network = (
            _SignalRegionShiftEstimator(
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

        if self.signal_region_shift_network is not None:
            nn.init.xavier_uniform_(
                self.signal_region_shift_network.hidden.weight,
                gain=gain,
            )
            nn.init.uniform_(self.signal_region_shift_network.hidden.bias, a=-0.3, b=0.3)
            nn.init.xavier_uniform_(
                self.signal_region_shift_network.output.weight,
                gain=gain,
            )
            nn.init.uniform_(self.signal_region_shift_network.output.bias, a=-0.3, b=0.3)

        self.nuisance_calculation.initialize_parameters(gain)

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

    def _signal_region_shift(self, sr_events: torch.Tensor) -> torch.Tensor:
        if self.signal_region_shift_network is None:
            raise RuntimeError("The denominator has no signal-region shift estimator.")
        return self.signal_region_shift_network(sr_events).squeeze(-1)

    def forward(
        self,
        data: _PreparedTrainingData,
        profiler: Optional[TrainingProfiler] = None,
    ) -> torch.Tensor:
        profile_region = profiler.region if profiler is not None else nullcontext
        with profile_region("training/signal_region_shift"):
            signal_hypothesis_sr_shift = (
                None
                if self.signal_region_shift_network is None
                else self._signal_region_shift(data.sr_events)
            )
        with profile_region("training/nuisance_theta"):
            nuisance_estimates = self.nuisance_calculation.evaluate(data=data.nuisance_data)
        return self._assemble_loss(
            signal_hypothesis_sr_shift=signal_hypothesis_sr_shift,
            nuisance_estimates=nuisance_estimates,
            data=data,
        )

    @staticmethod
    def _assemble_loss(
        *,
        signal_hypothesis_sr_shift: Optional[torch.Tensor],
        nuisance_estimates: NuisanceEvaluation,
        data: _PreparedTrainingData,
    ) -> torch.Tensor:
        """Assemble the negative log-likelihood used in the paper.

        For a signal-region event x, theta(x) the learned detector nuisance,
        and f(x), g(x) the learned signal hypothesis sifts for a, b in the
        signal region.

        The SR loss term:
            in the numerator (signal hypotehsis) is:

                sum_sr (
                    N_a_sr exp(f(x)) (1 + theta(x)) + N_b_sr exp(g(x)) (1 - theta(x))
                )
                - sum_a_sr f(x) - sum_b_sr g(x)

            in the denominator (null hypothesis) is:

                sum_sr (
                    N_a_sr (1 + theta(x)) + N_b_sr (1 - theta(x))
                )

        The CR term, in both cases:

            sum_cr (
                N_a_cr (1 + theta(x)) + N_b_cr (1 - theta(x))
            )
            - sum_a log(1 + theta(x)) - sum_b (1 - theta(x))

        """

        nuisance_sr_estimates = nuisance_estimates.nuisance_sr_values
        nuisance_cr_estimates = nuisance_estimates.nuisance_cr_values

        common_a_sr_nuisance_log_term = -torch.log1p(
            nuisance_sr_estimates[data.a_sr_mask]
        ).sum()
        common_b_sr_nuisance_log_term = -torch.log1p(
            -nuisance_sr_estimates[data.b_sr_mask]
        ).sum()

        cr_linear_nuisance_term = (
            data.nuisance_cr_coefficient
            * torch.dot(
                nuisance_cr_estimates,
                nuisance_estimates.nuisance_cr_a_weights
                + nuisance_estimates.nuisance_cr_b_weights,
            )
        )
        a_cr_log_term = -torch.dot(
            torch.log1p(nuisance_cr_estimates),
            nuisance_estimates.nuisance_cr_a_weights,
        )
        b_cr_log_term = -torch.dot(
            torch.log1p(-nuisance_cr_estimates),
            nuisance_estimates.nuisance_cr_b_weights,
        )
        cr_loss = (
            data.number_of_cr_events
            + cr_linear_nuisance_term
            + a_cr_log_term
            + b_cr_log_term
        )

        if signal_hypothesis_sr_shift is None:
            null_hypothesis_sr_loss = (
                data.N_sr
                + (data.n_a_sr_over_n_sr - data.n_b_sr_over_n_sr)
                * nuisance_sr_estimates.sum()
                + common_a_sr_nuisance_log_term
                + common_b_sr_nuisance_log_term
            )
            return null_hypothesis_sr_loss + cr_loss

        signal_region_shift = signal_hypothesis_sr_shift
        signal_hypothesis_a_sr_term = data.n_a_sr_over_n_sr * (
            1 + signal_region_shift
        )
        signal_hypothesis_b_sr_term = data.n_b_sr_over_n_sr * (
            1 - signal_region_shift
        )
        signal_hypothesis_sr_term = torch.addcmul(
            signal_hypothesis_a_sr_term + signal_hypothesis_b_sr_term,
            nuisance_sr_estimates,
            signal_hypothesis_a_sr_term - signal_hypothesis_b_sr_term,
        )
        signal_hypothesis_a_sr_f_log_term = -torch.log1p(
            signal_region_shift[data.a_sr_mask]
        ).sum()
        signal_hypothesis_b_sr_f_log_term = -torch.log1p(
            -signal_region_shift[data.b_sr_mask]
        ).sum()
        signal_hypothesis_sr_loss = (
            signal_hypothesis_sr_term.sum()
            + signal_hypothesis_a_sr_f_log_term
            + signal_hypothesis_b_sr_f_log_term
            + common_a_sr_nuisance_log_term
            + common_b_sr_nuisance_log_term
        )
        return signal_hypothesis_sr_loss + cr_loss

    def _prepare_training_data(self, data: DataBatch) -> _PreparedTrainingData:
        normalized_data, self._norm_factor = data.get_normalized()
        categories = DataSet.DataSetCategory
        normalized_a_sr = normalized_data.datasets[categories.A_SR]
        normalized_b_sr = normalized_data.datasets[categories.B_SR]
        normalized_a_cr = normalized_data.datasets[categories.A_CR]
        normalized_b_cr = normalized_data.datasets[categories.B_CR]
        a_sr = data.datasets[categories.A_SR]
        b_sr = data.datasets[categories.B_SR]
        a_cr = data.datasets[categories.A_CR]
        b_cr = data.datasets[categories.B_CR]

        N_sr = a_sr.n_samples + b_sr.n_samples
        N_cr = a_cr.n_samples + b_cr.n_samples
        if N_sr == 0:
            raise ValueError("Training requires at least one SR event.")
        if N_cr == 0:
            raise ValueError("Training requires at least one CR event.")

        normalized_sr = DataSet(
            np.concatenate((normalized_a_sr.events, normalized_b_sr.events))
        )
        raw_sr = DataSet(np.concatenate((a_sr.events, b_sr.events)))
        sr_data = torch.tensor(
            normalized_sr.events,
            dtype=self._dtype,
            device=self._device,
        )
        nuisance_data = self.nuisance_calculation.prepare(
            raw_sr=raw_sr,
            raw_a_cr=a_cr,
            raw_b_cr=b_cr,
            normalized_sr=normalized_sr,
            normalized_a_cr=normalized_a_cr,
            normalized_b_cr=normalized_b_cr,
        )
        a_sr_mask = torch.cat(
            (
                torch.ones(
                    a_sr.n_samples,
                    dtype=torch.bool,
                    device=self._device,
                ),
                torch.zeros(
                    b_sr.n_samples,
                    dtype=torch.bool,
                    device=self._device,
                ),
            )
        )
        return _PreparedTrainingData(
            sr_events=sr_data,
            nuisance_data=nuisance_data,
            a_sr_mask=a_sr_mask,
            b_sr_mask=~a_sr_mask,
            N_a_sr=a_sr.n_samples,
            N_b_sr=b_sr.n_samples,
            N_a_cr=a_cr.n_samples,
            N_b_cr=b_cr.n_samples,
            n_a_sr_over_n_sr=a_sr.n_samples / N_sr,
            n_b_sr_over_n_sr=b_sr.n_samples / N_sr,
            nuisance_cr_coefficient=(a_cr.n_samples - b_cr.n_samples)
            / N_cr,
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
        learning_rate = self._config.train__learning_rate + progress * (
            final_learning_rate - self._config.train__learning_rate
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
                self.nuisance_calculation.clamp_parameters()

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
        theta_sign: float,
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
            if self.signal_region_shift_network is None:
                signal_weight = x_tensor.new_ones((x_tensor.shape[0], 1))
            else:
                signal_region_shift = self.signal_region_shift_network(x_tensor)
                signal_weight = 1 + (
                    -signal_region_shift if secondary else signal_region_shift
                )
            if isinstance(self.nuisance_calculation, NeuralPerEventNuisanceEstimator):
                theta_estimate = self.nuisance_calculation.network(x_tensor).clamp(
                    min=-_SIGNAL_REGION_SHIFT_BOUND,
                    max=_SIGNAL_REGION_SHIFT_BOUND,
                ).unsqueeze(1)
            elif isinstance(self.nuisance_calculation, ScalarBinnedNuisanceEstimator):
                theta_estimate = self.nuisance_calculation._values(
                    self.nuisance_calculation._bin_indices(data)
                ).unsqueeze(1)
            else:
                theta_estimate = x_tensor.new_zeros((x_tensor.shape[0], 1))
            theta_term = torch.clamp(1 + theta_sign * theta_estimate, min=1e-12)
            predictions = signal_weight * theta_term
        return predictions.detach().cpu().numpy()

    def predict(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, secondary=False, theta_sign=1.0)

    def predict_secondary(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, secondary=True, theta_sign=-1.0)

    def predict_theta(self, data: DataSet) -> npt.NDArray:
        """Evaluate the configured nuisance function theta over a dataset."""
        self.eval()
        with torch.no_grad():
            if self._config.train__data_is_train_for_nuisances:
                if self.theta_network is not None:
                    if self._norm_factor is None:
                        raise RuntimeError(
                            "Cannot predict before the model has been fitted."
                        )
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
