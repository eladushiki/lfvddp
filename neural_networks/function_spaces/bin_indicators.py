"""Independent geometry and lookup for the binned function-space family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from neural_networks.function_spaces.base import (
    CoefficientTopology,
    EventInput,
    FunctionSpaceMetadata,
    FunctionSpaceRegularity,
    PerEventFunctionSpace,
    events_tensor,
    immutable_options,
    unexpected_construction_options,
)
from neural_networks.likelihood_parameterization import LIKELIHOOD_SHIFT_BOUND
from neural_networks.nuisance_contract import NuisanceCalculation
from train.function_space_config import FunctionSpaceFamily


@dataclass(frozen=True)
class BinIndicatorGeometry:
    """Fixed bin geometry owned by the function space, not the detector."""

    minima: tuple[float, ...]
    maxima: tuple[float, ...]
    number_of_bins: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.minima) == len(self.maxima) == len(self.number_of_bins)):
            raise ValueError(
                "Bin geometry minima, maxima, and number_of_bins must have equal lengths."
            )
        if not self.number_of_bins or any(number <= 0 for number in self.number_of_bins):
            raise ValueError("Bin geometry must define a positive number of bins per dimension.")
        if any(minimum >= maximum for minimum, maximum in zip(self.minima, self.maxima)):
            raise ValueError("Each bin geometry minimum must be smaller than its maximum.")

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "BinIndicatorGeometry":
        def values(value: Any, converter: Callable[[Any], Any]) -> tuple[Any, ...]:
            if isinstance(value, (str, bytes)):
                raise ValueError("Bin geometry options must be sequences.")
            try:
                return tuple(converter(item) for item in value)
            except TypeError as error:
                raise ValueError("Bin geometry options must be sequences.") from error

        try:
            return cls(
                minima=values(options["minima"], float),
                maxima=values(options["maxima"], float),
                number_of_bins=values(options["number_of_bins"], int),
            )
        except KeyError as error:
            raise ValueError(f"Missing bin geometry option {error.args[0]!r}.") from error

    @property
    def edges(self) -> tuple[npt.NDArray[np.float64], ...]:
        return tuple(
            np.linspace(minimum, maximum, number + 1)
            for minimum, maximum, number in zip(
                self.minima, self.maxima, self.number_of_bins
            )
        )

    def indices(self, events: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Return clipped zero-based indices for a numeric event matrix."""

        array = np.asarray(events)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[1] != len(self.number_of_bins):
            raise ValueError("Events must have one column per configured bin-geometry dimension.")

        indices = [
            np.clip(
                np.digitize(array[:, dimension], edges) - 1,
                a_min=0,
                a_max=number - 1,
            )
            for dimension, (edges, number) in enumerate(zip(self.edges, self.number_of_bins))
        ]
        return np.column_stack(indices).astype(np.int64, copy=False)


class BinIndicatorFunction(PerEventFunctionSpace):
    """Piecewise-constant factorized-bin function with fixed geometry."""

    family = FunctionSpaceFamily.BIN_INDICATORS
    metadata = FunctionSpaceMetadata(
        regularity=FunctionSpaceRegularity.PIECEWISE_CONSTANT,
        coefficient_topology=CoefficientTopology.FACTORIZED_MARGINAL_BINS,
    )
    feature_count = 0

    def __init__(
        self,
        geometry: BinIndicatorGeometry,
        *,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: Optional[torch.device | str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.options = immutable_options(
            options
            if options is not None
            else {
                "minima": list(geometry.minima),
                "maxima": list(geometry.maxima),
                "number_of_bins": list(geometry.number_of_bins),
            }
        )
        self._factor_deltas = nn.ParameterDict(
            {
                f"dimension_{index}": nn.Parameter(
                    torch.empty(number_of_bins, dtype=dtype, device=device)
                )
                for index, number_of_bins in enumerate(geometry.number_of_bins)
            }
        )
        self.feature_count = sum(geometry.number_of_bins)
        for index, edges in enumerate(geometry.edges):
            self.register_buffer(
                f"_edges_{index}",
                torch.tensor(edges, dtype=dtype, device=device),
                persistent=False,
            )

    def prediction_grid_edges(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Expose the family-owned bin edges required for prediction grids."""

        return self.geometry.edges

    @property
    def input_dimension(self) -> int:
        return len(self.geometry.number_of_bins)

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        BinIndicatorGeometry.from_options(options)

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> "BinIndicatorFunction":
        """Construct the binned function from its configuration envelope."""

        dtype = construction.pop("dtype", torch.get_default_dtype())
        device = construction.pop("device", None)
        construction.pop("output_dimension", None)
        geometry = construction.pop("geometry", None)
        unexpected_construction_options(cls.family, construction)
        return cls(
            BinIndicatorGeometry.from_options(options) if geometry is None else geometry,
            dtype=dtype,
            device=device,
            options=options,
        )

    def build_nuisance_calculation(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "NuisanceCalculation":
        """Adapt this family for its compact scalar control-region reduction."""

        from neural_networks.nuisance_calculation import BinnedNuisanceCalculation

        return BinnedNuisanceCalculation(
            dtype=dtype,
            device=device,
            function_space=self,
        )

    def bin_indices(self, events: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return self.geometry.indices(events)

    def evaluate(self, events: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Return fixed bin indices for geometry consumers."""

        return self.bin_indices(events)

    def _tensor_bin_indices(self, events: EventInput) -> torch.Tensor:
        values = events_tensor(
            events,
            self.input_dimension,
            dtype=self._factor_deltas["dimension_0"].dtype,
            device=self._factor_deltas["dimension_0"].device,
        )
        return torch.stack(
            tuple(
                torch.bucketize(
                    values[:, dimension],
                    getattr(self, f"_edges_{dimension}"),
                    right=True,
                ).sub(1).clamp_(min=0, max=number_of_bins - 1)
                for dimension, number_of_bins in enumerate(self.geometry.number_of_bins)
            ),
            dim=1,
        )

    def values_from_indices(self, bin_indices: torch.Tensor) -> torch.Tensor:
        """Return the bounded factorized value for one index tuple per event."""

        values = torch.ones(
            bin_indices.shape[0],
            dtype=self._factor_deltas["dimension_0"].dtype,
            device=self._factor_deltas["dimension_0"].device,
        )
        for dimension in range(self.input_dimension):
            values = values * torch.index_select(
                self._factor_deltas[f"dimension_{dimension}"],
                0,
                bin_indices[:, dimension],
            )
        return values.clamp(
            min=-LIKELIHOOD_SHIFT_BOUND,
            max=LIKELIHOOD_SHIFT_BOUND,
        )

    def forward(self, events: EventInput) -> torch.Tensor:
        return self.values_from_indices(self._tensor_bin_indices(events)).unsqueeze(-1)

    def initialize_parameters(self, _gain: float) -> None:
        for parameter in self._factor_deltas.values():
            nn.init.normal_(parameter, mean=0.0, std=1e-3)

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            for parameter in self._factor_deltas.values():
                parameter.clamp_(
                    min=-LIKELIHOOD_SHIFT_BOUND,
                    max=LIKELIHOOD_SHIFT_BOUND,
                )
