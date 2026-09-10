from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
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
from neural_networks.function_spaces import (
    AdaptiveNeuralFunction,
    create_function_space,
    initialize_function_space_parameters,
)
from neural_networks.nuisance_calculation import (
    BlankNuisanceEstimator,
    NeuralPerEventNuisanceEstimator,
    NuisanceEvaluation,
    PreparedNuisanceData,
    ScalarBinnedNuisanceEstimator,
    WeightedNuisanceValues,
    build_nuisance_calculation,
)
from train.function_space_config import FunctionSpaceFamily, RoleState
from neural_networks.utils import (
    ContextedModel,
    save_model_parameters_outcome,
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
    nuisance_data: PreparedNuisanceData
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


# Keep the historical private name for checkpoint and adapter compatibility.
# The implementation is shared with the nuisance neural family.
_SignalRegionShiftEstimator = AdaptiveNeuralFunction


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
        resolver = getattr(self._config, "resolve_function_space_config", None)
        if not callable(resolver):
            raise TypeError("DifferentiatingModel requires a TrainConfig resolver.")
        self._function_space_config = resolver()
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
        return build_nuisance_calculation(
            config=self._config,
            dtype=self._dtype,
            device=self._device,
            resolved_config=self._function_space_config,
        )

    def _build_signal_hypothesis_estimator(self) -> None:
        if not self._is_numerator:
            self.signal_region_shift_network = None
            return

        spec = self._function_space_config.f
        if spec.state is RoleState.DISABLED:
            raise ValueError("The f function-space role cannot be disabled.")
        construction = {}
        options = spec.options
        if spec.family is FunctionSpaceFamily.ADAPTIVE_NEURAL:
            if "input_dimension" not in options:
                input_dimension = getattr(self._config, "train__nn_input_dimension", None)
                if input_dimension is not None:
                    construction["input_dimension"] = input_dimension
            if not ({"hidden_size", "hidden_layer_nodes"} & set(options)):
                hidden_size = getattr(self._config, "train__nn_inner_layer_nodes", None)
                if hidden_size is not None:
                    construction["hidden_size"] = hidden_size
        if "output_dimension" not in options:
            construction["output_dimension"] = getattr(
                self._config, "train__nn_output_dimension", 1
            )
        estimator = create_function_space(
            "f",
            spec,
            dtype=self._dtype,
            device=self._assigned_device,
            **construction,
        )
        if not isinstance(estimator, nn.Module):
            raise ValueError(
                f"f function-space family {spec.family.value!r} is not trainable; "
                "only adaptive_neural is supported by the LFVDDP model in this slice."
            )
        self.signal_region_shift_network = estimator

    def _initialize_parameters(self) -> None:
        """
        Create newly initialized weights matching the training strategy.
        This is the single source of truth for weight initialization.
        Assumes 2-layer network (1 hidden layer).
        """
        # Use Xavier uniform with configurable gain for weight initialization
        gain = self._config.train__nn_xavier_gain

        if self.signal_region_shift_network is not None:
            initialize_function_space_parameters(
                self.signal_region_shift_network,
                gain,
            )

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
            nuisance_estimates = (
                None
                if isinstance(self.nuisance_calculation, BlankNuisanceEstimator)
                else self.nuisance_calculation.evaluate(data=data.nuisance_data)
            )
        return self._assemble_loss(
            signal_hypothesis_sr_shift=signal_hypothesis_sr_shift,
            nuisance_estimates=nuisance_estimates,
            data=data,
        )

    @staticmethod
    def _weighted_sum(nuisance_values: WeightedNuisanceValues) -> torch.Tensor:
        """Sum per-event values or compact values with event multiplicities."""

        if nuisance_values.weights is None:
            return nuisance_values.values.sum()
        return torch.dot(nuisance_values.values, nuisance_values.weights)

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

        a_sr_log_term = -torch.log1p(
            signal_region_shift[:number_of_a_sr_events]
        ).sum()
        b_sr_log_term = -torch.log1p(
            -signal_region_shift[number_of_a_sr_events:]
        ).sum()
        return a_sr_log_term, b_sr_log_term

    @staticmethod
    def _assemble_loss_without_nuisance(
        *,
        signal_hypothesis_sr_shift: Optional[torch.Tensor],
        data: _PreparedTrainingData,
    ) -> torch.Tensor:
        """Assemble the same loss without constructing zero nuisance arithmetic."""

        if signal_hypothesis_sr_shift is None:
            return data.sr_events.new_tensor(data.N_sr) + data.number_of_cr_events

        signal_region_shift = signal_hypothesis_sr_shift
        signal_hypothesis_sr_integral = data.N_sr + DifferentiatingModel._scaled_term(
            data.sr_category_imbalance,
            signal_region_shift.sum,
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
        )
        return signal_hypothesis_sr_loss + data.number_of_cr_events

    @staticmethod
    def _assemble_loss(
        *,
        signal_hypothesis_sr_shift: Optional[torch.Tensor],
        nuisance_estimates: Optional[NuisanceEvaluation],
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

        if nuisance_estimates is None:
            return DifferentiatingModel._assemble_loss_without_nuisance(
                signal_hypothesis_sr_shift=signal_hypothesis_sr_shift,
                data=data,
            )

        nuisance_sr_estimates = nuisance_estimates.nuisance_sr_values
        common_a_sr_nuisance_log_term = -torch.log1p(
            nuisance_sr_estimates[: data.N_a_sr]
        ).sum()
        common_b_sr_nuisance_log_term = -torch.log1p(
            -nuisance_sr_estimates[data.N_a_sr :]
        ).sum()

        cr_linear_nuisance_term = DifferentiatingModel._scaled_term(
            data.nuisance_cr_coefficient,
            lambda: (
                DifferentiatingModel._weighted_sum(nuisance_estimates.nuisance_cr_a)
                + DifferentiatingModel._weighted_sum(
                    nuisance_estimates.nuisance_cr_b
                )
            ),
        )
        a_cr_log_term = -DifferentiatingModel._weighted_sum(
            WeightedNuisanceValues(
                torch.log1p(nuisance_estimates.nuisance_cr_a.values),
                nuisance_estimates.nuisance_cr_a.weights,
            )
        )
        b_cr_log_term = -DifferentiatingModel._weighted_sum(
            WeightedNuisanceValues(
                torch.log1p(-nuisance_estimates.nuisance_cr_b.values),
                nuisance_estimates.nuisance_cr_b.weights,
            )
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
                + DifferentiatingModel._scaled_term(
                    data.sr_category_imbalance,
                    nuisance_sr_estimates.sum,
                )
                + common_a_sr_nuisance_log_term
                + common_b_sr_nuisance_log_term
            )
            return null_hypothesis_sr_loss + cr_loss

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
        if self._norm_factor is None:
            normalized_data, self._norm_factor = data.get_normalized()
        else:
            normalized_data = DataBatch(
                (dataset / self._norm_factor, parameters)
                for dataset, parameters in data
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
        return _PreparedTrainingData(
            sr_events=sr_data,
            nuisance_data=nuisance_data,
            N_a_sr=a_sr.n_samples,
            N_b_sr=b_sr.n_samples,
            N_a_cr=a_cr.n_samples,
            N_b_cr=b_cr.n_samples,
            sr_category_imbalance=(
                a_sr.n_samples / N_sr - b_sr.n_samples / N_sr
            ),
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

    @staticmethod
    def _checkpoint_value(value):
        """Convert frozen config values into deterministic JSON-compatible values."""

        if isinstance(value, Mapping):
            return {
                str(key): DifferentiatingModel._checkpoint_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [DifferentiatingModel._checkpoint_value(item) for item in value]
        if hasattr(value, "value"):
            return DifferentiatingModel._checkpoint_value(value.value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise TypeError(
            f"Unsupported checkpoint metadata value {type(value).__name__}."
        )

    def checkpoint_metadata(self) -> dict:
        """Describe the model shape/configuration without changing state_dict keys."""

        resolved = self._function_space_config
        compatibility = {
            "model_name": self._name,
            "is_numerator": self._is_numerator,
            "backend": self._checkpoint_value(resolved.backend),
            "f": {
                "family": self._checkpoint_value(resolved.f.family),
                "state": self._checkpoint_value(resolved.f.state),
                "options": self._checkpoint_value(resolved.f.options),
            },
            "nuisance": {
                "family": self._checkpoint_value(resolved.nuisance.family),
                "state": self._checkpoint_value(resolved.nuisance.state),
                "options": self._checkpoint_value(resolved.nuisance.options),
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "format_version": 1,
            **compatibility,
            "config_fingerprint": fingerprint,
            "normalization_factor": (
                None
                if self._norm_factor is None
                else {
                    "factors": self._checkpoint_value(self._norm_factor._factors),
                    "offsets": self._checkpoint_value(self._norm_factor._offsets),
                }
            ),
        }

    def _validate_checkpoint_metadata(self, checkpoint_path, metadata: dict) -> None:
        expected = self.checkpoint_metadata()
        # Normalization is runtime data, not model compatibility.  It is restored
        # below after the structural configuration has been checked.  Unknown
        # metadata is ignored so sidecars can gain optional fields safely.
        expected.pop("normalization_factor")
        actual = {key: metadata.get(key) for key in expected}
        if actual != expected:
            differences = []
            for key in sorted(set(expected) | set(actual)):
                if expected.get(key) != actual.get(key):
                    differences.append(key)
            changed = ", ".join(differences) or "unknown metadata"
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is incompatible with {self._name}: "
                f"{changed} differs. Refusing to load before state_dict validation."
            )

    def _restore_checkpoint_normalization(self, checkpoint_path, metadata: dict) -> None:
        normalization = metadata.get("normalization_factor")
        if normalization is None:
            return
        if not isinstance(normalization, dict) or not {
            "factors", "offsets"
        } <= set(normalization):
            raise RuntimeError(
                f"Checkpoint metadata {checkpoint_path} has an invalid normalization_factor."
            )
        factors = normalization["factors"]
        offsets = normalization["offsets"]
        if not isinstance(factors, dict) or not isinstance(offsets, dict):
            raise RuntimeError(
                f"Checkpoint metadata {checkpoint_path} has invalid normalization mappings."
            )
        try:
            self._norm_factor = ShiftAndNormalizationFactor(
                {str(key): float(value) for key, value in factors.items()},
                {str(key): float(value) for key, value in offsets.items()},
            )
        except (TypeError, ValueError, AssertionError) as error:
            raise RuntimeError(
                f"Checkpoint metadata {checkpoint_path} has an invalid normalization_factor."
            ) from error

    def _load_training_checkpoint_if_requested(
        self, optimizer: Optional[optim.Optimizer]
    ) -> int:
        checkpoint_result = find_latest_training_checkpoint(
            self._context, self._name, warn_missing=False
        )
        if checkpoint_result is None:
            return 0

        checkpoint_path, checkpoint = checkpoint_result
        metadata = load_checkpoint_metadata(checkpoint_path)
        if metadata is not None:
            self._validate_checkpoint_metadata(checkpoint_path, metadata)
            self._restore_checkpoint_normalization(checkpoint_path, metadata)
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
        self.train()
        optimizer = self.configure_optimizers()

        target_epochs = self._config.train__epochs
        start_epoch = self._load_training_checkpoint_if_requested(optimizer)
        # A new checkpoint carries the original normalization factor.  Prepare
        # data only after loading so continuation uses that exact transform.
        training_data = self._prepare_training_data(data)
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
                                metadata=self.checkpoint_metadata(),
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

    def _nuisance_prediction(self, data: DataSet) -> torch.Tensor:
        """Evaluate the configured nuisance estimator for prediction data."""
        if isinstance(self.nuisance_calculation, NeuralPerEventNuisanceEstimator):
            if self._norm_factor is None:
                raise RuntimeError("Cannot predict before the model has been fitted.")
            normalized_data = data / self._norm_factor
            normalized_events = torch.tensor(
                normalized_data.events,
                dtype=self._dtype,
                device=self._device,
            )
            nuisance_values = self.nuisance_calculation.network(normalized_events)
            return nuisance_values.squeeze(-1) if nuisance_values.ndim == 2 else nuisance_values
        if isinstance(self.nuisance_calculation, ScalarBinnedNuisanceEstimator):
            return self.nuisance_calculation._values(
                self.nuisance_calculation._bin_indices(data)
            )
        return torch.zeros(data.n_samples, dtype=self._dtype, device=self._device)

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
