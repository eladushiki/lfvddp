"""Independent geometry and lookup for the binned function-space family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

import numpy as np
import numpy.typing as npt
import torch

from neural_networks.function_spaces.base import (
    CoefficientTopology,
    FunctionSpaceMetadata,
    FunctionSpaceRegularity,
    immutable_options,
    unexpected_construction_options,
)
from train.function_space_config import FunctionSpaceFamily

if TYPE_CHECKING:
    from neural_networks.nuisance_calculation import NuisanceCalculation


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


class BinIndicatorFunction:
    """Piecewise-constant lookup with immutable family-specific geometry."""

    family = FunctionSpaceFamily.BIN_INDICATORS
    metadata = FunctionSpaceMetadata(
        regularity=FunctionSpaceRegularity.PIECEWISE_CONSTANT,
        coefficient_topology=CoefficientTopology.FACTORIZED_MARGINAL_BINS,
    )
    feature_count = 0

    def __init__(
        self,
        geometry: BinIndicatorGeometry,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
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

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> "BinIndicatorFunction":
        """Construct the binned lookup from its configuration envelope."""

        construction.pop("dtype", None)
        construction.pop("device", None)
        construction.pop("output_dimension", None)
        geometry = construction.pop("geometry", None)
        unexpected_construction_options(cls.family, construction)
        if geometry is None:
            return cls(BinIndicatorGeometry.from_options(options), options=options)
        return cls(geometry=geometry, options=options)

    def build_nuisance_calculation(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "NuisanceCalculation":
        """Adapt this binned lookup for compact scalar nuisance evaluation."""

        from neural_networks.nuisance_calculation import ScalarBinnedNuisanceEstimator

        return ScalarBinnedNuisanceEstimator(
            dtype=dtype,
            device=device,
            bin_lookup=self,
        )

    def bin_indices(self, events: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return self.geometry.indices(events)

    def evaluate(self, events: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return self.bin_indices(events)

    def initialize_parameters(self, gain: float) -> None:
        del gain
