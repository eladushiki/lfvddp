"""Cubic B-spline function-space family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from data_tools.data_utils import ShiftAndNormalizationFactor

from neural_networks.function_spaces.base import (
    CoefficientTopology,
    DeterministicFeatureFunction,
    EventInput,
    FunctionSpaceMetadata,
    FunctionSpaceRegularity,
    dimensions,
    events_tensor,
    number_sequence,
    require_options,
)
from train.function_space_config import FunctionSpaceFamily


CUBIC_BSPLINE_DEGREE = 3


@dataclass(frozen=True)
class CubicBSplineGeometry:
    """Clamped cubic knot vectors, one immutable vector per input dimension."""

    knots: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.knots:
            raise ValueError("Cubic B-spline geometry requires at least one knot vector.")
        for knots in self.knots:
            if len(knots) < 8 or any(left > right for left, right in zip(knots, knots[1:])):
                raise ValueError("Each cubic B-spline knot vector must be nondecreasing and valid.")
            if knots[0] == knots[-1]:
                raise ValueError("Each cubic B-spline knot vector must span a nonzero domain.")

    @property
    def feature_counts(self) -> tuple[int, ...]:
        return tuple(len(knots) - CUBIC_BSPLINE_DEGREE - 1 for knots in self.knots)

    @property
    def feature_count(self) -> int:
        return sum(self.feature_counts)

    @classmethod
    def from_breakpoints(cls, value: Any) -> "CubicBSplineGeometry":
        return cls(tuple(cls.clamped_knot_vector(knots) for knots in dimensions(value, "knots")))

    @staticmethod
    def clamped_knot_vector(knots: Sequence[float]) -> tuple[float, ...]:
        values = number_sequence(knots, "knots")
        if len(values) >= 8 and values[0] < values[-1] and all(
            left <= right for left, right in zip(values, values[1:])
        ):
            left_count = next((index for index, item in enumerate(values) if item != values[0]), len(values))
            right_count = next((index for index, item in enumerate(reversed(values)) if item != values[-1]), len(values))
            if left_count >= 4 and right_count >= 4:
                return values
        if len(values) < 2 or any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError(
                "Cubic B-spline knots must be strictly increasing breakpoints or a valid full vector."
            )
        return (values[0],) * 4 + values[1:-1] + (values[-1],) * 4


class CubicBSplineFunction(DeterministicFeatureFunction):
    """Additive cubic B-spline features with fixed, clamped knots."""

    family = FunctionSpaceFamily.CUBIC_BSPLINE
    metadata = FunctionSpaceMetadata(
        FunctionSpaceRegularity.CUBIC_SPLINE,
        CoefficientTopology.LINEAR_COEFFICIENTS,
    )

    def __init__(
        self,
        geometry: CubicBSplineGeometry,
        *,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: Optional[torch.device | str] = None,
        output_dimension: int = 1,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.geometry = geometry
        self._input_dimension = len(geometry.knots)
        super().__init__(
            geometry.feature_count,
            output_dimension=output_dimension,
            dtype=dtype,
            device=device,
            options=options,
        )
        for index, knots in enumerate(geometry.knots):
            self.register_buffer(f"_knots_{index}", torch.tensor(knots, dtype=dtype, device=device))

    @classmethod
    def geometry_from_options(cls, options: Mapping[str, Any]) -> CubicBSplineGeometry:
        require_options(options, "cubic_bspline", ("knots",))
        return CubicBSplineGeometry.from_breakpoints(options["knots"])

    def _knot_vector(self, dimension: int) -> torch.Tensor:
        knots = self._buffers[f"_knots_{dimension}"]
        assert isinstance(knots, torch.Tensor)
        return knots

    def normalize_input_geometry(
        self,
        normalization_factor: ShiftAndNormalizationFactor,
        observable_names: tuple[str, ...],
    ) -> None:
        """Derive normalized knots from the immutable physical knot vectors."""

        if len(observable_names) != self.input_dimension:
            raise ValueError("Function-space geometry does not match the observable dimension.")
        with torch.no_grad():
            for dimension, (name, knots) in enumerate(zip(observable_names, self.geometry.knots)):
                normalized = normalization_factor.normalize_values(
                    np.asarray(knots)[:, None], (name,)
                )[:, 0]
                buffer = self._knot_vector(dimension)
                buffer.copy_(torch.as_tensor(normalized, dtype=buffer.dtype, device=buffer.device))

    @staticmethod
    def _basis(values: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
        count = knots.numel() - CUBIC_BSPLINE_DEGREE - 1
        bases = [
            ((values >= knots[index]) & (values < knots[index + 1])).to(values.dtype)
            for index in range(count + CUBIC_BSPLINE_DEGREE)
        ]
        bases[-1] = ((values >= knots[-2]) & (values <= knots[-1])).to(values.dtype)
        for order in range(1, CUBIC_BSPLINE_DEGREE + 1):
            next_bases = []
            for index in range(count + CUBIC_BSPLINE_DEGREE - order):
                left_denominator = knots[index + order] - knots[index]
                right_denominator = knots[index + order + 1] - knots[index + 1]
                left = torch.zeros_like(values) if left_denominator == 0 else (values - knots[index]) / left_denominator * bases[index]
                right = torch.zeros_like(values) if right_denominator == 0 else (knots[index + order + 1] - values) / right_denominator * bases[index + 1]
                next_bases.append(left + right)
            bases = next_bases
        result = torch.stack(bases[:count], dim=1)
        right_boundary = values == knots[-1]
        if torch.any(right_boundary):
            result = torch.where(
                right_boundary[:, None],
                torch.nn.functional.one_hot(
                    torch.full_like(values, count - 1, dtype=torch.long), num_classes=count
                ).to(values.dtype),
                result,
            )
        return result

    def features(self, events: EventInput) -> torch.Tensor:
        values = events_tensor(
            events,
            self.input_dimension,
            dtype=self.coefficients.dtype,
            device=self.coefficients.device,
        )
        return torch.cat(
            tuple(
                self._basis(values[:, dimension], self._knot_vector(dimension))
                for dimension in range(self.input_dimension)
            ),
            dim=1,
        )
