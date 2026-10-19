from collections.abc import Mapping
from dataclasses import dataclass
from logging import warning
from typing import Any

from train.function_space_config import (
    FunctionSpaceSpec,
    ResolvedFunctionSpaceConfig,
    TrainingBackend,
    resolve_dual_role_config,
)


@dataclass
class TrainConfig:
    """Training controls and canonical likelihood function-space specifications."""

    train__epochs: int
    train__number_of_epochs_for_checkpoint: int
    train__initialize_from_null: bool = False
    train__optimizer_strategy: str = "adam"
    train__learning_rate: float = 0.03
    train__final_learning_rate: float | None = None
    train__enable_progress_bar: bool = True
    train__profiling_enabled: bool = False
    train__profiling_warmup_epochs: int = 5
    train__profiling_active_epochs: int = 10
    train__function_space_xavier_gain: float = 1

    train__backend: TrainingBackend | str = TrainingBackend.LFVDDP
    train__f: FunctionSpaceSpec | Mapping[str, Any] | None = None
    train__nuisance: FunctionSpaceSpec | Mapping[str, Any] | None = None

    # NPLM-specific controls remain meaningful only for the NPLM backend.
    train__nn_weight_clipping: float = False
    train__nuisance_correction_types: str = ""
    train__shape_nuisance_std: float = 0
    train__shape_nuisance_mean: float = 0
    train__shape_nuisance_reference: float = 0
    train__norm_nuisance_std: float = 0
    train__norm_nuisance_mean: float = 0
    train__norm_nuisance_reference: float = 0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def train__function_space_config(self) -> ResolvedFunctionSpaceConfig:
        """Resolve immutable role specifications without retaining mutable cache state."""

        return self.resolve_function_space_config()

    def resolve_function_space_config(self) -> ResolvedFunctionSpaceConfig:
        return resolve_dual_role_config(
            backend=self.train__backend,
            f=self.train__f,
            nuisance=self.train__nuisance,
        )

    @property
    def train__is_nplm(self) -> bool:
        return self.train__function_space_config.backend is TrainingBackend.NPLM

    @property
    def train__adaptive_architecture(self) -> list[int]:
        """Return the canonical adaptive architecture, also used by the NPLM adapter."""

        from neural_networks.function_spaces import AdaptiveNeuralFunction

        return list(
            AdaptiveNeuralFunction.architecture_from_options(
                self.train__function_space_config.f.options,
                observable_count=self.detector__number_of_dimensions,
            )
        )

    def validate(self) -> None:
        resolved = self.resolve_function_space_config()
        from neural_networks.function_spaces import validate_function_space_specs

        validate_function_space_specs(resolved.f, resolved.nuisance)

        if (
            self.train__is_nplm
            and type(resolved.f.options["hidden_layer_nodes"]) is not int
        ):
            raise ValueError(
                "NPLM requires integer hidden_layer_nodes; list architectures are supported only by LFVDDP."
            )

        if type(self.train__initialize_from_null) is not bool:
            raise ValueError("train__initialize_from_null must be a boolean.")
        if self.train__initialize_from_null and (
            self.train__is_nplm or resolved.f.family != "adaptive_neural"
        ):
            raise ValueError(
                "Null initialization requires LFVDDP with adaptive_neural f."
            )

        if self.train__profiling_warmup_epochs < 0:
            raise ValueError("Profiling warmup epochs cannot be negative.")
        if self.train__profiling_active_epochs < 1:
            raise ValueError("Profiling active epochs must be positive.")
        if self.train__profiling_enabled and self.train__is_nplm:
            raise ValueError(
                "Training profiling is only supported for LFVDDP training."
            )

        if self.train__optimizer_strategy not in {"adam", "adam_lbfgs"}:
            raise ValueError("Unknown train__optimizer_strategy.")
        if self.train__epochs < 1:
            raise ValueError("Training epochs must be positive.")
        if self.train__optimizer_strategy == "adam_lbfgs":
            if self.train__is_nplm:
                raise ValueError("Adam/L-BFGS is only supported by LFVDDP.")
            if self.train__final_learning_rate is not None:
                raise ValueError("Adam/L-BFGS requires a constant Adam learning rate.")

        required_epochs = 100_000 if self.train__is_nplm else 500_000
        if (
            self.train__optimizer_strategy == "adam"
            and self.train__epochs < required_epochs
        ):
            warning("Training epochs may be insufficient for convergence.")

        if not self.train__is_nplm and self.train__nuisance_correction_types:
            raise ValueError(
                "train__nuisance_correction_types is only valid for the NPLM backend."
            )
