"""Shared contracts and primitives for role-neutral function spaces."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    TypeAlias,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from data_tools.data_utils import ShiftAndNormalizationFactor

from neural_networks.likelihood_parameterization import smoothly_bounded_likelihood_shift


EventInput: TypeAlias = torch.Tensor | npt.ArrayLike


@runtime_checkable
class FunctionSpace(Protocol):
    """One parameterized likelihood-shift function on normalized events."""

    family: str
    options: Mapping[str, Any]
    feature_count: int

    def forward(self, events: EventInput) -> torch.Tensor:
        ...

    def initialize_parameters(self, gain: float) -> None:
        ...

    @classmethod
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> "FunctionSpace":
        ...

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        """Validate this family's configuration-owned geometry."""
        ...

    def normalize_input_geometry(
        self,
        normalization_factor: ShiftAndNormalizationFactor,
        observable_names: Iterable[str],
    ) -> None:
        """Map physical configuration geometry into model-input coordinates."""
        ...

    def statistical_degrees_of_freedom(self) -> int | None:
        """Return the fixed hypothesis-space dimension, when defined."""
        ...


def unexpected_construction_options(
    family: str,
    construction: Mapping[str, Any],
) -> None:
    """Reject construction arguments that a family did not consume."""

    if construction:
        names = ", ".join(sorted(construction))
        raise TypeError(f"Unexpected {family} construction option(s): {names}.")


def immutable_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy nested options into a read-only mapping owned by the function space."""

    def freeze(value: Any) -> Any:
        if isinstance(value, Mapping):
            return MappingProxyType({deepcopy(key): freeze(item) for key, item in value.items()})
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        return deepcopy(value)

    return MappingProxyType({key: freeze(value) for key, value in options.items()})


def require_options(options: Mapping[str, Any], family: str, names: Iterable[str]) -> None:
    """Reject incomplete geometry with a family-specific error."""

    missing = [name for name in names if name not in options]
    if missing:
        raise ValueError(f"{family} requires option(s): {', '.join(missing)}.")


def scalar_output_dimension(options: Mapping[str, Any], family: str) -> int:
    """Enforce the scalar likelihood-shift contract for every family."""

    output_dimension = options.get("output_dimension", 1)
    if output_dimension != 1:
        raise ValueError(f"{family} must have output_dimension equal to 1.")
    return 1


def number_sequence(value: Any, name: str) -> tuple[float, ...]:
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


def dimensions(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    """Interpret a flat sequence as one dimension and nesting as many dimensions."""

    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be numeric geometry.")
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return (number_sequence(value, name),)
    if array.ndim == 1:
        return (number_sequence(value, name),)
    if array.ndim == 2:
        return tuple(number_sequence(row, name) for row in value)
    raise ValueError(f"{name} must be one- or two-dimensional geometry.")


def events_tensor(
    events: EventInput,
    input_dimension: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a two-dimensional event tensor with the configured width."""

    if isinstance(events, torch.Tensor):
        tensor = events
    else:
        tensor = torch.as_tensor(events)
    if tensor.ndim == 1:
        if input_dimension != 1:
            raise ValueError(
                f"Expected events with shape (n, {input_dimension}), got {tuple(tensor.shape)}."
            )
        tensor = tensor[:, None]
    if tensor.ndim != 2 or tensor.shape[1] != input_dimension:
        raise ValueError(
            f"Expected events with shape (n, {input_dimension}), got {tuple(tensor.shape)}."
        )
    return tensor.to(device=device, dtype=dtype)


class PerEventFunctionSpace(nn.Module):
    """A trainable function space evaluated independently for each event."""

    def normalize_input_geometry(
        self,
        normalization_factor: ShiftAndNormalizationFactor,
        observable_names: Iterable[str],
    ) -> None:
        """Accept normalized inputs; geometry-free families need no adjustment."""

        del normalization_factor, observable_names

    def statistical_degrees_of_freedom(self) -> int | None:
        """Return no analytic rank for adaptive feature-search spaces."""

        return None

    @classmethod
    def analytic_degrees_of_freedom(cls, options: Mapping[str, Any]) -> int | None:
        """Return the configured fixed-space dimension without constructing a module."""

        del options
        return None

class DeterministicFeatureFunction(PerEventFunctionSpace):
    """Common linear-coefficient topology for fixed feature geometries."""

    @classmethod
    def construction_kwargs(
        cls,
        options: Mapping[str, Any],
        construction: dict[str, Any],
    ) -> dict[str, Any]:
        """Consume the construction envelope shared by fixed families."""

        if construction.pop("geometry", None) is not None:
            raise TypeError(f"{cls.family} does not accept geometry overrides.")
        result = {
            "dtype": construction.pop("dtype", torch.get_default_dtype()),
            "device": construction.pop("device", None),
            "output_dimension": scalar_output_dimension(options, cls.family),
        }
        unexpected_construction_options(cls.family, construction)
        return result

    @classmethod
    def geometry_from_options(cls, options: Mapping[str, Any]) -> Any:
        """Construct this fixed family's immutable feature geometry."""

        raise NotImplementedError(f"{cls.__name__} must define its feature geometry.")

    @classmethod
    def _statistical_constraint_dimension_for_geometry(cls, geometry: Any) -> int:
        del geometry
        return 0

    @classmethod
    def analytic_degrees_of_freedom(cls, options: Mapping[str, Any]) -> int:
        geometry = cls.geometry_from_options(options)
        output_dimension = scalar_output_dimension(options, cls.family)
        degrees_of_freedom = (
            geometry.feature_count * output_dimension
            - cls._statistical_constraint_dimension_for_geometry(geometry)
        )
        if degrees_of_freedom < 0:
            raise ValueError("Function-space constraints exceed trainable parameters.")
        return degrees_of_freedom

    @classmethod
    def validate_options(cls, options: Mapping[str, Any]) -> None:
        """Validate configuration by constructing the family-owned geometry."""

        cls.geometry_from_options(options)
        scalar_output_dimension(options, cls.family)

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **construction: Any) -> "FunctionSpace":
        """Build any fixed feature family through its common construction path."""

        return cls(
            cls.geometry_from_options(options),
            options=options,
            **cls.construction_kwargs(options, construction),
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
        if output_dimension != 1:
            raise ValueError("Function-space output_dimension must equal 1.")
        self.feature_count = int(feature_count)
        self.output_dimension = int(output_dimension)
        self.options = immutable_options(options or {})
        self.coefficients = nn.Parameter(
            torch.zeros(self.feature_count, self.output_dimension, dtype=dtype, device=device)
        )

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    def _linear_evaluation(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.coefficients

    def forward(self, events: EventInput) -> torch.Tensor:
        return smoothly_bounded_likelihood_shift(
            self._linear_evaluation(self.features(events))
        )

    def statistical_degrees_of_freedom(self) -> int:
        """Return this family's independent, constrained coefficient count."""

        return self.analytic_degrees_of_freedom(
            {**self.options, "output_dimension": self.output_dimension}
        )

    def _statistical_constraint_dimension(self) -> int:
        """Return fixed dependencies and observed-count constraints in this family."""

        return self._statistical_constraint_dimension_for_geometry(self.geometry)

    def initialize_parameters(self, gain: float) -> None:
        nn.init.xavier_uniform_(self.coefficients, gain=gain)
