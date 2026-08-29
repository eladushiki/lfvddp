from dataclasses import dataclass
from logging import warning
from typing import List, Optional, Tuple

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
    
    train__nn_xavier_gain: float = 4
    train__learning_rate: float = 0.001  # optimizer learning rate
    train__final_learning_rate: Optional[float] = None
    train__enable_progress_bar: bool = True
    train__lfvnn_max_cpu_threads: Optional[int] = None
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

    def _validate_nuisance_configuration(self) -> None:
        has_binning = any(
            parameter is not None
            for parameter in (
                self.train__nuisance_binning_minima,
                self.train__nuisance_binning_maxima,
                self.train__nuisance_binning_number_of_bins,
            )
        )
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
            if not all(
                parameter is not None
                for parameter in (
                    self.train__nuisance_binning_minima,
                    self.train__nuisance_binning_maxima,
                    self.train__nuisance_binning_number_of_bins,
                )
            ):
                raise ValueError(
                    "Binned nuisance configuration requires minima, maxima, and number of bins."
                )

    def configure_nuisance_binning(self, number_of_dimensions: int) -> None:
        """Normalize scalar binning parameters after detector dimensions are known."""
        if self.train__nuisance_is_neural_network:
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
        if self.train__nuisance_is_neural_network:
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
        self._validate_nuisance_configuration()

        if self.train__profiling_warmup_epochs < 0:
            raise ValueError("Profiling warmup epochs cannot be negative.")
        if self.train__profiling_active_epochs < 1:
            raise ValueError("Profiling active epochs must be positive.")
        if self.train__profiling_enabled and self.train__like_NPLM:
            raise ValueError(
                "Training profiling is only supported for LFVNN training."
            )
        if self.train__lfvnn_max_cpu_threads is not None and (
            isinstance(self.train__lfvnn_max_cpu_threads, bool)
            or not isinstance(self.train__lfvnn_max_cpu_threads, int)
            or self.train__lfvnn_max_cpu_threads < 1
        ):
            raise ValueError(
                "train__lfvnn_max_cpu_threads must be a positive integer or null."
            )

        if self.train__epochs < 1e5 and self.train__like_NPLM or \
                self.train__epochs < 5e5 and not self.train__like_NPLM:
            warning("Training epochs not sufficient, train may not converge")

        if not self.train__like_NPLM and \
                (self.train__nuisance_correction_types != "" or self.train__data_is_train_for_nuisances):
            warning("You probably meant to mimic LFVNN, but it does not deal with nuisances.")
