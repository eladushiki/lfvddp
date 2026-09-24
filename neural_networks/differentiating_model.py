from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from logging import info
from time import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from torch import nn, optim
from tqdm.auto import tqdm

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet, ShiftAndNormalizationFactor
from data_tools.detector.detector_config import DetectorConfig
from data_tools.detector.detector_effect import DetectorEffect
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.function_spaces import create_function_space
from neural_networks.utils import (
    ContextedModel,
    save_model_parameters_outcome,
)
from train.checkpoint_metadata import (
    build_checkpoint_metadata,
    normalization_from_checkpoint_metadata,
    validate_checkpoint_metadata,
)
from train.checkpoints import (
    find_latest_training_checkpoint,
    load_checkpoint_metadata,
    save_training_checkpoint,
)
from train.train_config import TrainConfig
from train.training_profiler import TrainingProfiler

LFVNN_DTYPE = torch.float64


@dataclass(frozen=True)
class _PreparedTrainingData:
    sr_events: torch.Tensor
    nuisance_events: torch.Tensor
    N_a_sr: int
    N_b_sr: int
    N_a_cr: int
    N_b_cr: int
    sr_category_imbalance: float
    nuisance_cr_coefficient: float

    @property
    def N_sr(self) -> int:
        return self.N_a_sr + self.N_b_sr

    @property
    def number_of_cr_events(self) -> int:
        return self.N_a_cr + self.N_b_cr


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
        if not isinstance(context.config, TrainConfig):
            raise TypeError("DifferentiatingModel requires a TrainConfig.")
        self._config: TrainConfig = context.config
        self._detector_effect = detector_effect
        self._is_numerator = is_numerator
        self._name = name
        self._dtype = dtype
        self._assigned_device = torch.device(device)
        self._function_space_config = self._config.resolve_function_space_config()
        self._norm_factor = None
        self._function_spaces_constructed = False
        self.signal_region_shift_network: Optional[nn.Module] = None
        self.nuisance_function_space: Optional[nn.Module] = None
        self._continuation_checkpoint_checked = False
        self._continuation_checkpoint: Optional[tuple] = None
        self._training_history = defaultdict(list)
        self._epochs_executed = 0
        self._best_model_state_dict: Optional[dict[str, torch.Tensor]] = None
        self._best_loss: Optional[float] = None
        self._best_epoch: Optional[int] = None

    @property
    def _device(self) -> torch.device:
        return self._assigned_device

    def _construct_function_spaces(
        self,
        normalization_factor: ShiftAndNormalizationFactor,
        observable_names: tuple[str, ...],
    ) -> None:
        """Construct both role modules once in the pooled model coordinates."""

        if self._is_numerator:
            self.signal_region_shift_network = create_function_space(
                self._function_space_config.f,
                dtype=self._dtype,
                device=self._device,
                normalization_factor=normalization_factor,
                observable_names=observable_names,
            )
        nuisance_spec = self._function_space_config.nuisance
        if nuisance_spec is not None:
            self.nuisance_function_space = create_function_space(
                nuisance_spec,
                dtype=self._dtype,
                device=self._device,
                normalization_factor=normalization_factor,
                observable_names=observable_names,
            )
        self._initialize_parameters()
        self.to(self._assigned_device)

    def _ensure_function_spaces_constructed(
        self,
        observable_names: tuple[str, ...],
    ) -> None:
        """Construct spaces once with the pooled map that defines model inputs."""

        if self._function_spaces_constructed:
            return
        if self._norm_factor is None:
            raise RuntimeError(
                "Cannot construct function spaces before data normalization."
            )
        self._construct_function_spaces(self._norm_factor, observable_names)
        self._function_spaces_constructed = True

    def _initialize_parameters(self) -> None:
        """
        Create newly initialized weights matching the training strategy.
        This is the single source of truth for weight initialization.
        Assumes 2-layer network (1 hidden layer).
        """
        # Use Xavier uniform with configurable gain for weight initialization
        gain = self._config.train__function_space_xavier_gain

        if self.signal_region_shift_network is not None:
            self.signal_region_shift_network.initialize_parameters(gain)

        if self.nuisance_function_space is not None:
            self.nuisance_function_space.initialize_parameters(gain)

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
        if not self._is_numerator:
            raise RuntimeError("The denominator has no signal-region shift estimator.")
        assert self.signal_region_shift_network is not None
        return self.signal_region_shift_network(sr_events).squeeze(-1)

    def forward(
        self,
        data: _PreparedTrainingData,
        profiler: Optional[TrainingProfiler] = None,
    ) -> torch.Tensor:
        profile_region = profiler.region if profiler is not None else nullcontext
        with profile_region("training/signal_region_shift"):
            signal_hypothesis_sr_shift = (
                self._signal_region_shift(data.sr_events)
                if self._is_numerator
                else data.sr_events.new_zeros(data.N_sr)
            )
        with profile_region("training/nuisance_theta"):
            nuisance_estimates = self._nuisance_values(data)
        return self._assemble_loss(
            signal_hypothesis_sr_shift=signal_hypothesis_sr_shift,
            nuisance_estimates=nuisance_estimates,
            data=data,
        )

    def _nuisance_values(
        self, data: _PreparedTrainingData
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate one shared nuisance space for all A/B and SR/CR event groups."""

        if self.nuisance_function_space is None:
            values = data.nuisance_events.new_zeros(data.nuisance_events.shape[0])
        else:
            values = self.nuisance_function_space(data.nuisance_events).squeeze(-1)
        return tuple(
            values.split((data.N_a_sr, data.N_b_sr, data.N_a_cr, data.N_b_cr))
        )

    @staticmethod
    def _scaled_term(
        coefficient: float,
        term: Callable[[], torch.Tensor],
    ) -> Union[float, torch.Tensor]:
        """Evaluate a tensor term only when its exact scalar coefficient is nonzero."""

        if coefficient == 0.0:
            return 0.0
        return coefficient * term()

    @staticmethod
    def _signal_shift_log_terms(
        signal_region_shift: torch.Tensor,
        number_of_a_sr_events: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the signal-shift log terms shared by both nuisance paths."""

        a_sr_log_term = -torch.log1p(signal_region_shift[:number_of_a_sr_events]).sum()
        b_sr_log_term = -torch.log1p(-signal_region_shift[number_of_a_sr_events:]).sum()
        return a_sr_log_term, b_sr_log_term

    @staticmethod
    def _assemble_loss(
        *,
        signal_hypothesis_sr_shift: torch.Tensor,
        nuisance_estimates: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        data: _PreparedTrainingData,
    ) -> torch.Tensor:
        """Assemble the negative log-likelihood used in the paper.

        For a signal-region event x, theta(x) is the learned detector nuisance
        and f(x) is the single learned signal shift. The A and B signal weights
        reciprocate as 1 + f(x) and 1 - f(x), respectively.

        The SR loss term:
            in the numerator (signal hypotehsis) is:

                sum_sr (
                    a (1 + f(x)) (1 + theta(x))
                    + b (1 - f(x)) (1 - theta(x))
                )
                - sum_a_sr [log(1 + f(x)) + log(1 + theta(x))]
                - sum_b_sr [log(1 - f(x)) + log(1 - theta(x))]

            in the denominator (null hypothesis) is:

                sum_sr (
                    N_a_sr (1 + theta(x)) + N_b_sr (1 - theta(x))
                )

        The CR term, in both cases:

            sum_cr (
                N_a_cr (1 + theta(x)) + N_b_cr (1 - theta(x))
            )
            - sum_a log(1 + theta(x)) - sum_b log(1 - theta(x))

        """

        a_sr_nuisance, b_sr_nuisance, a_cr_nuisance, b_cr_nuisance = nuisance_estimates
        nuisance_sr_estimates = torch.cat((a_sr_nuisance, b_sr_nuisance))
        common_a_sr_nuisance_log_term = -torch.log1p(a_sr_nuisance).sum()
        common_b_sr_nuisance_log_term = -torch.log1p(-b_sr_nuisance).sum()

        cr_linear_nuisance_term = DifferentiatingModel._scaled_term(
            data.nuisance_cr_coefficient,
            lambda: (
                a_cr_nuisance.sum() + b_cr_nuisance.sum()
            ),
        )
        a_cr_log_term = -torch.log1p(a_cr_nuisance).sum()
        b_cr_log_term = -torch.log1p(-b_cr_nuisance).sum()
        cr_loss = (
            data.number_of_cr_events
            + cr_linear_nuisance_term
            + a_cr_log_term
            + b_cr_log_term
        )

        signal_region_shift = signal_hypothesis_sr_shift
        signal_hypothesis_sr_integral = (
            data.N_sr
            + DifferentiatingModel._scaled_term(
                data.sr_category_imbalance,
                lambda: signal_region_shift.sum() + nuisance_sr_estimates.sum(),
            )
            + torch.dot(signal_region_shift, nuisance_sr_estimates)
        )
        (
            signal_hypothesis_a_sr_f_log_term,
            signal_hypothesis_b_sr_f_log_term,
        ) = DifferentiatingModel._signal_shift_log_terms(
            signal_region_shift,
            data.N_a_sr,
        )
        signal_hypothesis_sr_loss = (
            signal_hypothesis_sr_integral
            + signal_hypothesis_a_sr_f_log_term
            + signal_hypothesis_b_sr_f_log_term
            + common_a_sr_nuisance_log_term
            + common_b_sr_nuisance_log_term
        )
        return signal_hypothesis_sr_loss + cr_loss

    def _prepare_training_data(self, data: DataBatch) -> _PreparedTrainingData:
        self._restore_checkpoint_normalization_if_requested()
        if self._norm_factor is None:
            normalized_data, self._norm_factor = data.get_normalized()
        else:
            normalized_data = DataBatch(
                (dataset / self._norm_factor, parameters)
                for dataset, parameters in data
            )
        self._ensure_function_spaces_constructed(
            data.unified_data.observable_names
        )
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

        sr_data = torch.tensor(
            np.concatenate((normalized_a_sr.events, normalized_b_sr.events)),
            dtype=self._dtype,
            device=self._device,
        )
        nuisance_events = torch.tensor(
            np.concatenate(
                (
                    normalized_a_sr.events,
                    normalized_b_sr.events,
                    normalized_a_cr.events,
                    normalized_b_cr.events,
                )
            ),
            dtype=self._dtype,
            device=self._device,
        )
        return _PreparedTrainingData(
            sr_events=sr_data,
            nuisance_events=nuisance_events,
            N_a_sr=a_sr.n_samples,
            N_b_sr=b_sr.n_samples,
            N_a_cr=a_cr.n_samples,
            N_b_cr=b_cr.n_samples,
            sr_category_imbalance=(a_sr.n_samples / N_sr - b_sr.n_samples / N_sr),
            nuisance_cr_coefficient=(a_cr.n_samples - b_cr.n_samples) / N_cr,
        )

    def _log(self, epoch: int, loss: torch.Tensor) -> None:
        self._training_history[HistoryKeys.LOSS.value].append(
            float(loss.detach().cpu())
        )
        self._training_history[HistoryKeys.EPOCH.value].append(epoch)

    def _model_state_snapshot(self) -> dict[str, torch.Tensor]:
        """Return an independent copy of the current model configuration."""

        return {
            name: value.detach().clone() for name, value in self.state_dict().items()
        }

    def _record_best_model_state(
        self,
        loss: torch.Tensor,
        epoch: int,
        model_state: dict[str, torch.Tensor],
    ) -> None:
        """Keep the lowest-loss configuration evaluated during optimization."""

        loss_value = float(loss.detach().cpu())
        if self._best_loss is None or loss_value < self._best_loss:
            self._best_loss = loss_value
            self._best_epoch = epoch
            self._best_model_state_dict = model_state

    def _restore_best_model_state(self) -> None:
        """Restore the configuration associated with the lowest observed loss."""

        if self._best_model_state_dict is not None:
            self.load_state_dict(self._best_model_state_dict)

    @property
    def minimum_loss(self) -> float:
        """Return the loss of the model configuration retained after training."""

        if self._best_loss is None:
            raise RuntimeError("Minimum loss is unavailable before model training.")
        return self._best_loss

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

    def _continuation_checkpoint_if_requested(self) -> Optional[tuple]:
        """Locate a continuation checkpoint at most once per model lifecycle."""

        if not self._continuation_checkpoint_checked:
            self._continuation_checkpoint = find_latest_training_checkpoint(
                self._context, self._name, warn_missing=False
            )
            self._continuation_checkpoint_checked = True
        return self._continuation_checkpoint

    def _restore_checkpoint_normalization_if_requested(self) -> None:
        """Restore coordinate metadata before modules are constructed."""

        if self._norm_factor is not None:
            return
        checkpoint_result = self._continuation_checkpoint_if_requested()
        if checkpoint_result is None:
            return
        checkpoint_path, _ = checkpoint_result
        metadata = load_checkpoint_metadata(checkpoint_path)
        restored_normalization = normalization_from_checkpoint_metadata(
            checkpoint_path, metadata
        )
        if restored_normalization is not None:
            self._norm_factor = restored_normalization

    def _load_training_checkpoint_if_requested(
        self, optimizer: Optional[optim.Optimizer]
    ) -> int:
        checkpoint_result = self._continuation_checkpoint_if_requested()
        if checkpoint_result is None:
            return 0

        checkpoint_path, checkpoint = checkpoint_result
        metadata = load_checkpoint_metadata(checkpoint_path)
        validate_checkpoint_metadata(
            checkpoint_path=checkpoint_path,
            model_name=self._name,
            expected=build_checkpoint_metadata(
                model_name=self._name,
                is_numerator=self._is_numerator,
                resolved_config=self._function_space_config,
                normalization_factor=self._norm_factor,
            ),
            actual=metadata,
        )
        self.load_state_dict(checkpoint["model_state_dict"], strict=True)
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
        best_state = checkpoint.get("best_model_state_dict")
        if best_state is not None:
            self._best_model_state_dict = {
                name: value.detach().clone() for name, value in best_state.items()
            }
            self._best_loss = float(checkpoint["best_loss"])
            self._best_epoch = int(checkpoint["best_epoch"])
        else:
            # Checkpoints written before minimum-state tracking can resume from
            # their current configuration, but cannot reconstruct older states.
            self._best_model_state_dict = self._model_state_snapshot()
            losses = self._training_history.get(HistoryKeys.LOSS.value, [])
            self._best_loss = float(losses[-1]) if losses else None
            self._best_epoch = int(checkpoint.get("epoch", -1))
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
        return loss

    def fit(
        self,
        data: DataBatch,
    ) -> Dict[str, List[float]]:
        self.train()
        target_epochs = self._config.train__epochs
        # Construct spaces only after the pooled normalization map is known.
        training_data = self._prepare_training_data(data)
        optimizer = self.configure_optimizers()
        start_epoch = self._load_training_checkpoint_if_requested(optimizer)
        if start_epoch >= target_epochs:
            self._restore_best_model_state()
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
                    model_state_before_step = self._model_state_snapshot()
                    epoch_last_predictions = self._train_step(
                        optimizer=optimizer,
                        data=training_data,
                        profiler=profiler,
                    )
                    self._record_best_model_state(
                        epoch_last_predictions,
                        epoch,
                        model_state_before_step,
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
                                metadata=build_checkpoint_metadata(
                                    model_name=self._name,
                                    is_numerator=self._is_numerator,
                                    resolved_config=self._function_space_config,
                                    normalization_factor=self._norm_factor,
                                ),
                                best_model_state_dict=self._best_model_state_dict,
                                best_loss=self._best_loss,
                                best_epoch=self._best_epoch,
                            )
                profiler.step()

        self._restore_best_model_state()

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

    def _nuisance_prediction(self, data: DataSet) -> torch.Tensor:
        """Evaluate the configured nuisance estimator for prediction data."""
        if self._norm_factor is None:
            raise RuntimeError("Cannot predict before the model has been fitted.")
        self._ensure_function_spaces_constructed(data.observable_names)
        normalized_events = torch.as_tensor(
            (data / self._norm_factor).events,
            dtype=self._dtype,
            device=self._device,
        )
        if self.nuisance_function_space is None:
            return normalized_events.new_zeros(normalized_events.shape[0])
        return self.nuisance_function_space(normalized_events).squeeze(-1)

    def _predict_ndf(
        self,
        data: DataSet,
        secondary: bool,
        theta_sign: float,
    ) -> npt.NDArray:
        if self._norm_factor is None:
            raise RuntimeError("Cannot predict before the model has been fitted.")
        self._ensure_function_spaces_constructed(data.observable_names)
        normalized_data = data / self._norm_factor
        x_tensor = torch.tensor(
            normalized_data.events,
            dtype=self._dtype,
            device=self._device,
        )
        self.eval()
        with torch.no_grad():
            signal_region_shift = (
                self.signal_region_shift_network(x_tensor)
                if self.signal_region_shift_network is not None
                else x_tensor.new_zeros((x_tensor.shape[0], 1))
            )
            signal_weight = 1 + (
                -signal_region_shift if secondary else signal_region_shift
            )
            theta_estimate = self._nuisance_prediction(data).unsqueeze(1)
            theta_term = torch.clamp(1 + theta_sign * theta_estimate, min=1e-12)
            predictions = signal_weight * theta_term
        return predictions.detach().cpu().numpy()

    def predict(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, secondary=False, theta_sign=1.0)

    def predict_secondary(self, data: DataSet) -> npt.NDArray:
        return self._predict_ndf(data, secondary=True, theta_sign=-1.0)

    def predict_theta(self, data: DataSet) -> npt.NDArray:
        """Evaluate the configured nuisance estimator over a dataset."""
        self.eval()
        with torch.no_grad():
            predictions = self._nuisance_prediction(data).unsqueeze(1)
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

    # Module shape and physical geometry are determined by the pooled batch map.
    model._prepare_training_data(data)

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

    if model.has_trainable_parameters():
        final_loss = model.minimum_loss
    else:
        final_loss = min(model_history[HistoryKeys.LOSS.value])
    info(f"Minimum loss achieved: {final_loss:.6f}")

    save_model_parameters_outcome(context, model)

    return model, final_loss, model_history
