from dataclasses import dataclass, field
from logging import warning
from typing import Any, List, Mapping, Optional, Tuple

from train.function_space_config import (
    FunctionSpaceFamily,
    FunctionSpaceSpec,
    ResolvedFunctionSpaceConfig,
    TrainingBackend,
    resolve_dual_role_config,
)

import numpy as np
import numpy.typing as npt


@dataclass
class TrainConfig:
    
    ## Training parameters
    train__epochs: int
    train__number_of_epochs_for_checkpoint: int

    # NN parameters
    train__nn_inner_layer_nodes: int
    train__nn_input_dimension: Optional[int] = None
    @property
    def train__nn_output_dimension(self) -> int:
        return 1
    @property
    def train__nn_architecture(self) -> List[int]:
        return [self.train__nn_input_dimension, self.train__nn_inner_layer_nodes, self.train__nn_output_dimension]
    
    train__nn_xavier_gain: float = 1
    train__learning_rate: float = 0.03  # optimizer learning rate
    train__final_learning_rate: Optional[float] = None
    train__enable_progress_bar: bool = True
    # Opt-in CPU profiling. The warmup epochs are observed by the profiler but
    # omitted from its measurements; the following active epochs are recorded.
    train__profiling_enabled: bool = False
    train__profiling_warmup_epochs: int = 5
    train__profiling_active_epochs: int = 10
    
    train__like_NPLM: bool = False

    train__backend: TrainingBackend | str | None = None
    train__f: FunctionSpaceSpec | Mapping[str, Any] | None = None
    train__nuisance: FunctionSpaceSpec | Mapping[str, Any] | None = None
    train__resolved_function_space_config: Optional[ResolvedFunctionSpaceConfig] = field(
        default=None, init=False, repr=False
    )

    def _f_with_adaptive_defaults(
        self,
    ) -> FunctionSpaceSpec | Mapping[str, Any]:
        """Add configured adaptive-network dimensions to the canonical f specification."""
        assert self.train__f is not None
        if isinstance(self.train__f, FunctionSpaceSpec):
            family = self.train__f.family
            options = self.train__f.options
        else:
            try:
                family = FunctionSpaceFamily.from_value(self.train__f.get("family"))
            except ValueError:
                return self.train__f
            options = self.train__f.get("options", {})
            if options is None:
                options = {}
            if not isinstance(options, Mapping):
                return self.train__f

        if family is not FunctionSpaceFamily.ADAPTIVE_NEURAL:
            return self.train__f

        resolved_options = dict(options)
        if "input_dimension" not in resolved_options and self.train__nn_input_dimension is not None:
            resolved_options["input_dimension"] = self.train__nn_input_dimension
        if not ({"hidden_size", "hidden_layer_nodes"} & set(resolved_options)):
            resolved_options["hidden_size"] = self.train__nn_inner_layer_nodes

        if isinstance(self.train__f, FunctionSpaceSpec):
            return FunctionSpaceSpec(
                family=family,
                options=resolved_options,
                state=self.train__f.state,
            )
        return {**self.train__f, "options": resolved_options}

    def resolve_function_space_config(self) -> ResolvedFunctionSpaceConfig:
        if self.train__f is None:
            raise ValueError("train__f must define a canonical function-space mapping.")
        if self.train__nuisance is None:
            raise ValueError("train__nuisance must define a canonical function-space mapping.")
        self.train__resolved_function_space_config = resolve_dual_role_config(
            backend=self.train__backend,
            f=self._f_with_adaptive_defaults(),
            nuisance=self.train__nuisance,
        )
        return self.train__resolved_function_space_config

    @property
    def train__function_space_config(self) -> ResolvedFunctionSpaceConfig:
        """Return the cached canonical configuration used by training code."""
        if self.train__resolved_function_space_config is None:
            return self.resolve_function_space_config()
        return self.train__resolved_function_space_config

    @property
    def train__number_of_nuisance_parameters(self) -> int:
        nuisance = self.train__function_space_config.nuisance
        if not nuisance.enabled or nuisance.family is not FunctionSpaceFamily.BIN_INDICATORS:
            return 0
        return sum(nuisance.options["number_of_bins"])

    # NPLM PARAMETERS -- only relevant if train__like_NPLM is True
    train__nn_weight_clipping: float = False
    # Correction - what should be taken into account about the nuisance parameters?
    # - "SHAPE" - both normalization and shape uncertainties are considered
    # - "NORM" - only normalization uncertainties are considered
    # - "" - systematic uncertainties are neglected (simple NPLM is run - no Delta calculation and Tau is calculated without nuisance parameters)
    train__nuisance_correction_types: str = ""  # "SHAPE", "NORM" or "". Which compensations for uncertainties to use.

    # Recovery of nuisances parameters
    train__shape_nuisance_std: float = 0                # shape nuisance sigma
    train__shape_nuisance_mean: float = 0               # shape nuisance reference, in terms of std
    train__shape_nuisance_reference: float = 0          # norm nuisance reference, in terms of std
    
    train__norm_nuisance_std: float = 0                 # norm nuisance sigma
    train__norm_nuisance_mean: float = 0                # in terms of std
    train__norm_nuisance_reference: float = 0           # in terms of std

    @property
    def train__nn_significant_degrees_of_freedom(self) -> int:
        # Calculate total trainable parameters (weights + biases) in the dense NN.
        # This does not include learnable nuisance parameters that may appear.
        # For architecture [n0, n1, n2, ...], params = sum over layers i: (n[i] * n[i+1] + n[i+1])
        architecture = self.train__nn_architecture
        total_params = sum(
            architecture[i] * architecture[i+1] + architecture[i+1]
            for i in range(len(architecture) - 1)
        )
        return total_params - 1  # The substraction is due to the argument about another constraint on the DoF in our paper

    def __post_init__(self):
        self.validate()

    def validate(self):
        self.resolve_function_space_config()

        if self.train__profiling_warmup_epochs < 0:
            raise ValueError("Profiling warmup epochs cannot be negative.")
        if self.train__profiling_active_epochs < 1:
            raise ValueError("Profiling active epochs must be positive.")
        if self.train__profiling_enabled and self.train__like_NPLM:
            raise ValueError(
                "Training profiling is only supported for LFVNN training."
            )

        if self.train__epochs < 1e5 and self.train__like_NPLM or \
                self.train__epochs < 5e5 and not self.train__like_NPLM:
            warning("Training epochs not sufficient, train may not converge")

        if not self.train__like_NPLM and (
            self.train__nuisance_correction_types != ""
            or self.train__function_space_config.nuisance.enabled
        ):
            warning("You probably meant to mimic LFVNN, but it does not deal with nuisances.")
