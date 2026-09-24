"""Fixed Cartesian bin-indicator likelihood function space."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, Optional

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as functional

from data_tools.data_utils import ShiftAndNormalizationFactor
from neural_networks.function_spaces.base import (
    DeterministicFeatureFunction,
    EventInput,
    events_tensor,
)


@dataclass(frozen=True)
class BinIndicatorGeometry:
    """Immutable physical bin edges for one Cartesian detector grid."""

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
        def real_values(value: Any, name: str) -> tuple[float, ...]:
            if isinstance(value, (str, bytes)):
                raise ValueError(f"Bin geometry {name} must be a numeric sequence.")
            try:
                result = tuple(float(item) for item in value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Bin geometry {name} must be a numeric sequence."
                ) from error
            if not result or not all(np.isfinite(item) for item in result):
                raise ValueError(f"Bin geometry {name} must contain finite values.")
            return result

        def bin_counts(value: Any) -> tuple[int, ...]:
            values = real_values(value, "number_of_bins")
            if any(not item.is_integer() for item in values):
                raise ValueError("Bin geometry number_of_bins must contain integers.")
            return tuple(int(item) for item in values)

        try:
            return cls(
                minima=real_values(options["minima"], "minima"),
                maxima=real_values(options["maxima"], "maxima"),
                number_of_bins=bin_counts(options["number_of_bins"]),
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

    @property
    def feature_count(self) -> int:
        return prod(self.number_of_bins)


class BinIndicatorFunction(DeterministicFeatureFunction):
    """One bounded linear coefficient per Cartesian detector-bin cell."""

    family = "bin_indicators"

    def __init__(
        self,
        geometry: BinIndicatorGeometry,
        *,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: Optional[torch.device | str] = None,
        output_dimension: int = 1,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.geometry = geometry
        self._input_dimension = len(geometry.number_of_bins)
        super().__init__(
            geometry.feature_count,
            output_dimension=output_dimension,
            dtype=dtype,
            device=device,
            options=options,
        )
        for dimension, edges in enumerate(geometry.edges):
            self.register_buffer(
                f"_edges_{dimension}",
                torch.as_tensor(edges, dtype=dtype, device=device),
            )

    @classmethod
    def geometry_from_options(cls, options: Mapping[str, Any]) -> BinIndicatorGeometry:
        return BinIndicatorGeometry.from_options(options)

    def _edges(self, dimension: int) -> torch.Tensor:
        edges = self._buffers[f"_edges_{dimension}"]
        assert isinstance(edges, torch.Tensor)
        return edges

    def normalize_input_geometry(
        self,
        normalization_factor: ShiftAndNormalizationFactor,
        observable_names: tuple[str, ...],
    ) -> None:
        if len(observable_names) != self.input_dimension:
            raise ValueError("Function-space geometry does not match the observable dimension.")
        with torch.no_grad():
            for name, edges, dimension in zip(
                observable_names, self.geometry.edges, range(self.input_dimension)
            ):
                normalized = normalization_factor.normalize_values(edges[:, None], (name,))[:, 0]
                self._edges(dimension).copy_(
                    torch.as_tensor(
                        normalized,
                        dtype=self.coefficients.dtype,
                        device=self.coefficients.device,
                    )
                )

    def _flat_bin_indices(self, events: EventInput) -> torch.Tensor:
        values = events_tensor(
            events,
            self.input_dimension,
            dtype=self.coefficients.dtype,
            device=self.coefficients.device,
        )
        flat_indices = torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
        stride = 1
        for dimension in reversed(range(self.input_dimension)):
            indices = torch.bucketize(
                values[:, dimension].contiguous(), self._edges(dimension), right=True
            ).sub(1).clamp_(min=0, max=self.geometry.number_of_bins[dimension] - 1)
            flat_indices.add_(indices * stride)
            stride *= self.geometry.number_of_bins[dimension]
        return flat_indices

    def features(self, events: EventInput) -> torch.Tensor:
        return functional.one_hot(
            self._flat_bin_indices(events), num_classes=self.feature_count
        ).to(dtype=self.coefficients.dtype)

    @classmethod
    def _statistical_constraint_dimension_for_geometry(
        cls, geometry: BinIndicatorGeometry
    ) -> int:
        del cls, geometry
        return 1
