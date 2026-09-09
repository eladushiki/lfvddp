from dataclasses import dataclass, field
from logging import warning
from typing import Any, List, Mapping, Optional, Tuple

from train.function_space_config import (
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
    
    ## Training for nuisance parameters
    train__data_is_train_for_nuisances: bool = True     # Should the nuisance play a role of learnable NN parameters?
    train__nuisance_is_neural_network: bool = False
    train__nuisance_nn_inner_layer_nodes: Optional[int] = None
    train__nuisance_binning_minima: Optional[List[float]] = None
    train__nuisance_binning_maxima: Optional[List[float]] = None
    train__nuisance_binning_number_of_bins: Optional[List[int]] = None
    train__like_NPLM: bool = False  # Should we trian with NPLM's train_model and nuisance parameters? else, DDP's

    # Canonical Issue 018 configuration.  The role mappings use the same
    # {family, options, state} schema for f and nuisance.
    train__backend: Optional[str] = None
    train__function_space_backend: Optional[str] = None
    train__function_space: Optional[Mapping[str, Any]] = None
    train__f_function_space: Optional[Mapping[str, Any]] = None
    train__nuisance_function_space: Optional[Mapping[str, Any]] = None
    # Short aliases are accepted at the composition boundary, then resolved
    # into the role mappings above.  Runtime code must use the resolved object.
    train__f: Optional[Mapping[str, Any]] = None
    train__nuisance: Optional[Mapping[str, Any]] = None
    train__resolved_function_space_config: Optional[ResolvedFunctionSpaceConfig] = field(
        default=None, init=False, repr=False, compare=False
    )

    def resolve_function_space_config(self) -> ResolvedFunctionSpaceConfig:
        """Resolve canonical role objects and legacy fields exactly once."""
        backend_values = {
            name: value
            for name, value in (
                ("train__backend", self.train__backend),
                ("train__function_space_backend", self.train__function_space_backend),
            )
            if value is not None
        }
        if len({TrainingBackend.from_value(value) for value in backend_values.values()}) > 1:
            raise ValueError(
                "Conflicting training backend fields: "
                + ", ".join(f"{name}={value!r}" for name, value in backend_values.items())
                + "."
            )
        backend = next(iter(backend_values.values()), None)
        nested = self.train__function_space
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise ValueError("train__function_space must be a mapping.")
            unknown = set(nested) - {"backend", "f", "nuisance"}
            if unknown:
                raise ValueError(
                    "train__function_space has unknown field(s): "
                    + ", ".join(sorted(str(item) for item in unknown))
                    + "."
                )
            nested_backend = nested.get("backend")
            if backend is not None and nested_backend is not None and TrainingBackend.from_value(backend) is not TrainingBackend.from_value(nested_backend):
                raise ValueError("Conflicting training backend fields: train__function_space.backend and train__backend.")
            backend = backend if backend is not None else nested_backend

        if self.train__f_function_space is not None and self.train__f is not None:
            raise ValueError("Conflicting f function-space fields: train__f_function_space and train__f.")
        if self.train__nuisance_function_space is not None and self.train__nuisance is not None:
            raise ValueError(
                "Conflicting nuisance function-space fields: "
                "train__nuisance_function_space and train__nuisance."
            )
        f_config = (
            self.train__f_function_space
            if self.train__f_function_space is not None
            else self.train__f
        )
        nuisance_config = (
            self.train__nuisance_function_space
            if self.train__nuisance_function_space is not None
            else self.train__nuisance
        )
        explicit_canonical_roles = f_config is not None or nuisance_config is not None
        if nested is not None:
            if f_config is not None and nested.get("f") is not None:
                raise ValueError("Conflicting f function-space fields: train__function_space.f and role field.")
            if nuisance_config is not None and nested.get("nuisance") is not None:
                raise ValueError("Conflicting nuisance function-space fields: train__function_space.nuisance and role field.")
            f_config = f_config if f_config is not None else nested.get("f")
            nuisance_config = nuisance_config if nuisance_config is not None else nested.get("nuisance")
            explicit_canonical_roles = f_config is not None or nuisance_config is not None

        if nuisance_config is not None and (
            not self.train__data_is_train_for_nuisances
            or self.train__nuisance_is_neural_network
            or self.train__nuisance_nn_inner_layer_nodes is not None
            or any(
                value is not None
                for value in (
                    self.train__nuisance_binning_minima,
                    self.train__nuisance_binning_maxima,
                    self.train__nuisance_binning_number_of_bins,
                )
            )
        ):
            raise ValueError(
                "Canonical nuisance configuration conflicts with legacy nuisance fields; "
                "use either train__nuisance_function_space or the train__nuisance_* fields."
            )

        if nuisance_config is None and self.train__nuisance_is_neural_network:
            nuisance_config = {
                "family": "adaptive_neural",
                "options": {
                    "input_dimension": self.train__nn_input_dimension,
                    "hidden_layer_nodes": self.train__nuisance_nn_inner_layer_nodes,
                    "xavier_gain": self.train__nn_xavier_gain,
                },
            }

        legacy_f_options = {
            "input_dimension": self.train__nn_input_dimension,
            "hidden_layer_nodes": self.train__nn_inner_layer_nodes,
            "xavier_gain": self.train__nn_xavier_gain,
        }
        legacy_f_options = {key: value for key, value in legacy_f_options.items() if value is not None}
        legacy_nuisance_options = {
            "minima": self.train__nuisance_binning_minima,
            "maxima": self.train__nuisance_binning_maxima,
            "number_of_bins": self.train__nuisance_binning_number_of_bins,
        }
        legacy_nuisance_options = {
            key: value for key, value in legacy_nuisance_options.items() if value is not None
        }
        self.train__resolved_function_space_config = resolve_dual_role_config(
            backend=backend,
            f=f_config,
            nuisance=nuisance_config,
            legacy_f_options=legacy_f_options,
            legacy_nuisance_options=legacy_nuisance_options,
            legacy_nuisance_enabled=self.train__data_is_train_for_nuisances,
            legacy_like_nplm=self.train__like_NPLM,
            compatibility_source="canonical" if explicit_canonical_roles else "legacy",
            validate_legacy=False,
        )
        return self.train__resolved_function_space_config

    @property
    def train__function_space_config(self) -> ResolvedFunctionSpaceConfig:
        """Resolved dual-role configuration used by later model adapters."""
        if self.train__resolved_function_space_config is None:
            return self.resolve_function_space_config()
        return self.train__resolved_function_space_config

    @property
    def resolved_function_space_config(self) -> ResolvedFunctionSpaceConfig:
        return self.train__function_space_config

    @property
    def train__resolved_function_spaces(self) -> ResolvedFunctionSpaceConfig:
        return self.train__function_space_config

    @property
    def train__f_function_space_spec(self) -> FunctionSpaceSpec:
        return self.train__function_space_config.f

    @property
    def train__nuisance_function_space_spec(self) -> FunctionSpaceSpec:
        return self.train__function_space_config.nuisance

    def _validate_nuisance_configuration(self) -> None:
        if (
            self.train__nuisance_function_space is not None
            or self.train__nuisance is not None
            or isinstance(self.train__function_space, Mapping)
            and "nuisance" in self.train__function_space
        ):
            return
        binning_parameters = (
            self.train__nuisance_binning_minima,
            self.train__nuisance_binning_maxima,
            self.train__nuisance_binning_number_of_bins,
        )
        has_binning = any(parameter is not None for parameter in binning_parameters)
        has_complete_binning = all(
            parameter is not None for parameter in binning_parameters
        )
        if not self.train__data_is_train_for_nuisances:
            if has_binning and not has_complete_binning:
                raise ValueError(
                    "Nuisance binning configuration must define minima, maxima, "
                    "and number of bins together."
                )
            return

        if self.train__nuisance_is_neural_network:
            if has_binning:
                raise ValueError(
                    "Neural nuisance configuration must not define nuisance binning parameters."
                )
            if self.train__nuisance_nn_inner_layer_nodes is None:
                raise ValueError(
                    "Neural nuisance configuration requires train__nuisance_nn_inner_layer_nodes."
                )
        else:
            if self.train__nuisance_nn_inner_layer_nodes is not None:
                raise ValueError(
                    "Binned nuisance configuration must not define train__nuisance_nn_inner_layer_nodes."
                )
            if not has_complete_binning:
                raise ValueError(
                    "Binned nuisance configuration requires minima, maxima, and number of bins."
                )

    def configure_nuisance_binning(self, number_of_dimensions: int) -> None:
        """Normalize scalar binning parameters after detector dimensions are known."""
        if (
            self.train__nuisance_is_neural_network
            or self.train__nuisance_binning_minima is None
        ):
            return

        for parameter_name in (
            "train__nuisance_binning_minima",
            "train__nuisance_binning_maxima",
            "train__nuisance_binning_number_of_bins",
        ):
            parameter = getattr(self, parameter_name)
            if isinstance(parameter, (int, float)):
                setattr(self, parameter_name, [parameter] * number_of_dimensions)
            elif len(parameter) != number_of_dimensions:
                raise ValueError(
                    f"{parameter_name} length {len(parameter)} does not match detector dimensions {number_of_dimensions}."
                )

    def observable_bins(self, observable_name: str) -> Tuple[npt.NDArray, npt.NDArray]:
        """Return bin edges and centers for a scalar binned nuisance observable."""
        try:
            index = self.detector__detect_observable_names.index(observable_name)
        except ValueError as error:
            raise ValueError(
                f"Observable name {observable_name} not found in detector observable names "
                f"{self.detector__detect_observable_names}"
            ) from error

        bins_edges = np.linspace(
            self.train__nuisance_binning_minima[index],
            self.train__nuisance_binning_maxima[index],
            self.train__nuisance_binning_number_of_bins[index] + 1,
        )
        return bins_edges, 0.5 * (bins_edges[:-1] + bins_edges[1:])

    @property
    def train__number_of_nuisance_parameters(self) -> int:
        if (
            not self.train__data_is_train_for_nuisances
            or self.train__nuisance_is_neural_network
        ):
            return 0
        return sum(self.train__nuisance_binning_number_of_bins)
    
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
        self._validate_nuisance_configuration()

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

        if not self.train__like_NPLM and \
                (self.train__nuisance_correction_types != "" or self.train__data_is_train_for_nuisances):
            warning("You probably meant to mimic LFVNN, but it does not deal with nuisances.")
