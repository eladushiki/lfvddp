"""Role-neutral deterministic feature-space families.

Geometry is parsed once into immutable tuples.  The geometry tensors are
registered as buffers so ``Module.to`` moves them with the trainable
coefficients, while only the coefficient matrix is exposed as a parameter.
All families return a design matrix from :meth:`features` and its linear
coefficient evaluation from :meth:`evaluate`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from neural_networks.function_spaces.base import FunctionSpaceMetadata
from neural_networks.likelihood_parameterization import smoothly_bounded_likelihood_shift
from train.function_space_config import FunctionSpaceFamily


def _require_options(options: Mapping[str, Any], family: str, names: Iterable[str]) -> None:
    missing = [name for name in names if name not in options]
    if missing:
        raise ValueError(f"{family} requires option(s): {', '.join(missing)}.")


def _immutable_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy options so caller-owned geometry cannot alias a function space."""

    def freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({deepcopy(key): freeze(item) for key, item in value.items()})
        if isinstance(value, list):
            return tuple(freeze(item) for item in value)
        if isinstance(value, tuple):
            return tuple(freeze(item) for item in value)
        return deepcopy(value)

    return MappingProxyType({key: freeze(value) for key, value in options.items()})


def _number_sequence(value: Any, name: str) -> tuple[float, ...]:
    """Convert one numeric sequence and reject malformed geometry."""

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a numeric sequence.")
    try:
        values = tuple(float(item) for item in value) if not np.isscalar(value) else (float(value),)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric sequence.") from error
    if not values or not all(np.isfinite(item) for item in values):
        raise ValueError(f"{name} must contain at least one finite value.")
    return values


def _dimensions(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    """Interpret a flat sequence as one dimension, or nested sequences as many."""

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be numeric geometry.")
    try:
        array = np.asarray(value, dtype=object)
    except Exception as error:
        raise ValueError(f"{name} must be numeric geometry.") from error
    if array.ndim <= 1:
        return (_number_sequence(value, name),)
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    return tuple(_number_sequence(row, f"{name}[{index}]") for index, row in enumerate(value))


def _events_tensor(events: Any, input_dimension: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Convert events to the common ``(event, observable)`` tensor shape."""

    tensor = events if isinstance(events, torch.Tensor) else torch.as_tensor(events)
    if tensor.ndim == 1:
        if input_dimension != 1:
            raise ValueError(
                f"Expected events with {input_dimension} observables, got a one-dimensional array."
            )
        tensor = tensor[:, None]
    if tensor.ndim != 2 or tensor.shape[1] != input_dimension:
        raise ValueError(
            f"Expected events with shape (n, {input_dimension}), got {tuple(tensor.shape)}."
        )
    return tensor.to(device=device, dtype=dtype)


class DeterministicFeatureFunction(nn.Module):
    """Common linear-coefficient topology for fixed feature geometries."""

    metadata = FunctionSpaceMetadata(
        regularity="deterministic",
        coefficient_topology="linear_coefficients",
    )

    def __init__(
        self,
        feature_count: int,
        *,
        output_dimension: int = 1,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: Optional[torch.device | str] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        if feature_count <= 0:
            raise ValueError("A deterministic function space must have at least one feature.")
        if output_dimension <= 0:
            raise ValueError("output_dimension must be positive.")
        self.feature_count = int(feature_count)
        self.output_dimension = int(output_dimension)
        self.options = _immutable_options(options or {})
        self.coefficients = nn.Parameter(
            torch.zeros(self.feature_count, self.output_dimension, dtype=dtype, device=device)
        )

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    def _linear_evaluation(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.coefficients

    def forward(self, events: Any) -> torch.Tensor:
        # ``evaluate`` remains the linear design-space value used by rank and
        # coefficient-linearity calculations. Role adapters use this bounded
        # path so likelihood log terms never receive a value outside (-1, 1).
        return smoothly_bounded_likelihood_shift(self.evaluate(events))

    def evaluate(self, events: Any) -> torch.Tensor:
        return self._linear_evaluation(self.features(events))

    def feature_map(self, events: Any) -> torch.Tensor:
        """Descriptive alias for callers that use feature-map terminology."""

        return self.features(events)


@dataclass(frozen=True)
class CubicBSplineGeometry:
    """Clamped cubic B-spline knot vectors, one immutable vector per dimension."""

    knots: tuple[tuple[float, ...], ...]
    degree: int = 3

    def __post_init__(self) -> None:
        if self.degree != 3 or not self.knots:
            raise ValueError("Cubic B-spline geometry requires cubic knot vectors.")
        for knots in self.knots:
            if len(knots) < 8 or any(left > right for left, right in zip(knots, knots[1:])):
                raise ValueError("Each cubic B-spline knot vector must be nondecreasing and valid.")
            if knots[0] == knots[-1]:
                raise ValueError("Each cubic B-spline knot vector must span a nonzero domain.")

    @property
    def feature_counts(self) -> tuple[int, ...]:
        return tuple(len(knots) - self.degree - 1 for knots in self.knots)

    @property
    def feature_count(self) -> int:
        return sum(self.feature_counts)


def _clamped_knot_vector(knots: Sequence[float]) -> tuple[float, ...]:
    values = _number_sequence(knots, "knots")
    if len(values) >= 8 and values[0] < values[-1] and all(
        left <= right for left, right in zip(values, values[1:])
    ):
        # A full vector is recognized by clamped end multiplicity.  Otherwise
        # a monotonic input is treated as breakpoints below.
        left_count = next((i for i, item in enumerate(values) if item != values[0]), len(values))
        right_count = next((i for i, item in enumerate(reversed(values)) if item != values[-1]), len(values))
        if left_count >= 4 and right_count >= 4:
            if len(values) - 4 <= 0:
                raise ValueError("Cubic B-spline knot vector has no basis functions.")
            return values
    if len(values) < 2 or any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError("Cubic B-spline knots must be strictly increasing breakpoints or a valid full vector.")
    return (values[0],) * 4 + values[1:-1] + (values[-1],) * 4


def _spline_basis(values: torch.Tensor, knots: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """Evaluate a Cox-de Boor basis, including the right boundary exactly."""

    count = knots.numel() - degree - 1
    bases = [((values >= knots[index]) & (values < knots[index + 1])).to(values.dtype) for index in range(count + degree)]
    bases[-1] = ((values >= knots[-2]) & (values <= knots[-1])).to(values.dtype)
    for order in range(1, degree + 1):
        next_bases = []
        for index in range(count + degree - order):
            left_denominator = knots[index + order] - knots[index]
            right_denominator = knots[index + order + 1] - knots[index + 1]
            left = (values - knots[index]) / left_denominator * bases[index] if left_denominator > 0 else torch.zeros_like(values)
            right = (knots[index + order + 1] - values) / right_denominator * bases[index + 1] if right_denominator > 0 else torch.zeros_like(values)
            next_bases.append(left + right)
        bases = next_bases
    result = torch.stack(bases[:count], dim=1)
    # Cox-de Boor's half-open intervals omit the right endpoint; the clamped
    # spline convention assigns it to the final basis exactly.
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


class CubicBSplineFunction(DeterministicFeatureFunction):
    """Additive cubic B-spline feature map with fixed, clamped knots."""

    family = FunctionSpaceFamily.CUBIC_BSPLINE
    metadata = FunctionSpaceMetadata("cubic_spline", "linear_coefficients")

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
        super().__init__(geometry.feature_count, output_dimension=output_dimension, dtype=dtype, device=device, options=options)
        for index, knots in enumerate(geometry.knots):
            self.register_buffer(f"_knots_{index}", torch.tensor(knots, dtype=dtype, device=device))

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **construction: Any) -> "CubicBSplineFunction":
        _require_options(options, "cubic_bspline", ("knots",))
        dimensions = _dimensions(options["knots"], "knots")
        geometry = CubicBSplineGeometry(tuple(_clamped_knot_vector(item) for item in dimensions))
        return cls(geometry, options=options, **construction)

    def features(self, events: Any) -> torch.Tensor:
        values = _events_tensor(events, self.input_dimension, dtype=self.coefficients.dtype, device=self.coefficients.device)
        return torch.cat(tuple(_spline_basis(values[:, index], getattr(self, f"_knots_{index}")) for index in range(self.input_dimension)), dim=1)


@dataclass(frozen=True)
class OrthogonalPolynomialGeometry:
    basis: str
    maximum_degree: int
    domain: tuple[tuple[float, float], ...]

    @property
    def feature_count(self) -> int:
        return len(self.domain) * (self.maximum_degree + 1)


def _polynomial_domains(value: Any) -> tuple[tuple[float, float], ...]:
    dimensions = _dimensions(value, "domain")
    result = []
    for domain in dimensions:
        if len(domain) != 2 or domain[0] >= domain[1]:
            raise ValueError("Each polynomial domain must contain an increasing (minimum, maximum) pair.")
        result.append((domain[0], domain[1]))
    return tuple(result)


class OrthogonalPolynomialFunction(DeterministicFeatureFunction):
    """Legendre or Chebyshev additive polynomial feature map."""

    family = FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL

    def __init__(
        self,
        geometry: OrthogonalPolynomialGeometry,
        *,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: Optional[torch.device | str] = None,
        output_dimension: int = 1,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if geometry.basis not in {"legendre", "chebyshev"}:
            raise ValueError("Polynomial basis must be 'legendre' or 'chebyshev'.")
        self.geometry = geometry
        self._input_dimension = len(geometry.domain)
        super().__init__(geometry.feature_count, output_dimension=output_dimension, dtype=dtype, device=device, options=options)
        self.register_buffer("_domain", torch.tensor(geometry.domain, dtype=dtype, device=device))

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **construction: Any) -> "OrthogonalPolynomialFunction":
        _require_options(options, "orthogonal_polynomial", ("basis", "maximum_degree", "domain"))
        basis = str(options["basis"]).strip().lower()
        degree = options["maximum_degree"]
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or degree < 0:
            raise ValueError("maximum_degree must be a nonnegative integer.")
        geometry = OrthogonalPolynomialGeometry(basis, int(degree), _polynomial_domains(options["domain"]))
        return cls(geometry, options=options, **construction)

    def features(self, events: Any) -> torch.Tensor:
        values = _events_tensor(events, self.input_dimension, dtype=self.coefficients.dtype, device=self.coefficients.device)
        minimum, maximum = self._domain[:, 0], self._domain[:, 1]
        x = 2 * (values - minimum) / (maximum - minimum) - 1
        columns = []
        for dimension in range(self.input_dimension):
            current = torch.ones_like(x[:, dimension])
            columns.append(current)
            if self.geometry.maximum_degree >= 1:
                previous = current
                current = x[:, dimension]
                columns.append(current)
                for degree in range(2, self.geometry.maximum_degree + 1):
                    if self.geometry.basis == "legendre":
                        next_value = ((2 * degree - 1) * x[:, dimension] * current - (degree - 1) * previous) / degree
                    else:
                        next_value = 2 * x[:, dimension] * current - previous
                    columns.append(next_value)
                    previous, current = current, next_value
        return torch.stack(columns, dim=1)


@dataclass(frozen=True)
class CenterGeometry:
    centers: tuple[tuple[float, ...], ...]
    widths: tuple[tuple[float, ...], ...]

    @property
    def feature_count(self) -> int:
        return len(self.centers)

    @property
    def input_dimension(self) -> int:
        return len(self.centers[0])


def _center_geometry(options: Mapping[str, Any], family_name: str) -> CenterGeometry:
    raw_centers = options["centers"]
    if isinstance(raw_centers, (str, bytes)):
        raise ValueError(f"{family_name} centers must be numeric geometry.")
    center_array = np.asarray(raw_centers, dtype=object)
    if center_array.ndim == 0:
        centers = ((float(raw_centers),),)
    elif center_array.ndim == 1:
        # A flat list is the natural one-dimensional shorthand: one center
        # per entry, rather than one multidimensional center.
        centers = tuple((float(center),) for center in raw_centers)
    elif center_array.ndim == 2:
        centers = tuple(_number_sequence(center, "centers") for center in raw_centers)
    else:
        raise ValueError(f"{family_name} centers must be one- or two-dimensional.")
    if not centers or any(not all(np.isfinite(value) for value in center) for center in centers):
        raise ValueError(f"{family_name} centers must contain at least one finite value.")
    dimension = len(centers[0])
    if any(len(center) != dimension for center in centers):
        raise ValueError(f"{family_name} centers must have consistent dimensionality.")

    raw_widths = options["widths"]
    if isinstance(raw_widths, (str, bytes)):
        raise ValueError(f"{family_name} widths must be numeric geometry.")
    width_array = np.asarray(raw_widths, dtype=object)
    if width_array.ndim == 0:
        width_matrix = tuple((float(raw_widths),) * dimension for _ in centers)
    elif width_array.ndim == 1:
        width_values = _number_sequence(raw_widths, "widths")
        if len(width_values) == 1:
            width_matrix = tuple(width_values * dimension for _ in centers)
        elif len(width_values) == len(centers):
            width_matrix = tuple((width,) * dimension for width in width_values)
        elif len(centers) == 1 and len(width_values) == dimension:
            width_matrix = (width_values,)
        else:
            raise ValueError(f"{family_name} widths must be scalar, one per center, or one vector per center.")
    elif width_array.ndim == 2 and len(width_array) == len(centers):
        width_matrix = tuple(_number_sequence(width, "widths") for width in raw_widths)
        if any(len(width) != dimension for width in width_matrix):
            raise ValueError(f"{family_name} widths must match the center dimensionality.")
    else:
        raise ValueError(f"{family_name} widths must be scalar, one per center, or one vector per center.")
    if any(width <= 0 for widths in width_matrix for width in widths):
        raise ValueError(f"{family_name} widths must be strictly positive.")
    return CenterGeometry(tuple(centers), tuple(width_matrix))


def _raise_width_shape(family_name: str) -> tuple[float, ...]:
    raise ValueError(f"{family_name} widths must match the center dimensionality.")


class _CenteredFeatureFunction(DeterministicFeatureFunction):
    """Shared fixed-center geometry and tensor handling for sigmoid/RBF maps."""

    def __init__(
        self,
        geometry: CenterGeometry,
        *,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: Optional[torch.device | str] = None,
        output_dimension: int = 1,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.geometry = geometry
        self._input_dimension = geometry.input_dimension
        super().__init__(geometry.feature_count, output_dimension=output_dimension, dtype=dtype, device=device, options=options)
        self.register_buffer("_centers", torch.tensor(geometry.centers, dtype=dtype, device=device))
        self.register_buffer("_widths", torch.tensor(geometry.widths, dtype=dtype, device=device))

    def _centered_values(self, events: Any) -> torch.Tensor:
        values = _events_tensor(events, self.input_dimension, dtype=self.coefficients.dtype, device=self.coefficients.device)
        return (values[:, None, :] - self._centers[None, :, :]) / self._widths[None, :, :]


class FixedSigmoidFunction(_CenteredFeatureFunction):
    """Fixed-center sigmoid features with trainable output coefficients only."""

    family = FunctionSpaceFamily.FIXED_SIGMOID
    metadata = FunctionSpaceMetadata("smooth", "linear_coefficients")

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **construction: Any) -> "FixedSigmoidFunction":
        _require_options(options, "fixed_sigmoid", ("centers", "widths"))
        return cls(_center_geometry(options, "fixed_sigmoid"), options=options, **construction)

    def features(self, events: Any) -> torch.Tensor:
        return torch.sigmoid(self._centered_values(events)).prod(dim=2)


class GaussianRadialBasisFunction(_CenteredFeatureFunction):
    """Fixed-center Gaussian radial features with trainable output coefficients only."""

    family = FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS
    metadata = FunctionSpaceMetadata("smooth", "linear_coefficients")

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **construction: Any) -> "GaussianRadialBasisFunction":
        _require_options(options, "gaussian_radial_basis", ("centers", "widths"))
        return cls(_center_geometry(options, "gaussian_radial_basis"), options=options, **construction)

    def features(self, events: Any) -> torch.Tensor:
        return torch.exp(-0.5 * self._centered_values(events).square().sum(dim=2))


# Short names make registry and downstream imports readable without introducing
# a second implementation or a role-specific alias.
CubicBSpline = CubicBSplineFunction
OrthogonalPolynomial = OrthogonalPolynomialFunction
FixedSigmoid = FixedSigmoidFunction
GaussianRadialBasis = GaussianRadialBasisFunction

__all__ = [
    "CenterGeometry",
    "CubicBSpline",
    "CubicBSplineFunction",
    "CubicBSplineGeometry",
    "DeterministicFeatureFunction",
    "FixedSigmoid",
    "FixedSigmoidFunction",
    "GaussianRadialBasis",
    "GaussianRadialBasisFunction",
    "OrthogonalPolynomial",
    "OrthogonalPolynomialFunction",
    "OrthogonalPolynomialGeometry",
]
