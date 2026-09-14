"""Shared contracts and primitives for role-neutral function spaces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
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

from neural_networks.likelihood_parameterization import smoothly_bounded_likelihood_shift
from train.function_space_config import FunctionSpaceFamily

if TYPE_CHECKING:
    from neural_networks.nuisance_calculation import NuisanceCalculation


class FunctionSpaceRegularity(str, Enum):
    """Regularity categories declared by concrete function-space families."""

    ADAPTIVE = "adaptive"
    PIECEWISE_CONSTANT = "piecewise_constant"
    CUBIC_SPLINE = "cubic_spline"
    ORTHOGONAL_POLYNOMIAL = "orthogonal_polynomial"
    SMOOTH = "smooth"
    DETERMINISTIC = "deterministic"


class CoefficientTopology(str, Enum):
    """Coefficient-layout categories declared by concrete function-space families."""

    DENSE_TWO_LAYER = "dense_two_layer"
    FACTORIZED_MARGINAL_BINS = "factorized_marginal_bins"
    LINEAR_COEFFICIENTS = "linear_coefficients"


@dataclass(frozen=True)
class FunctionSpaceMetadata:
    """Structural metadata consumed by adapters and diagnostics."""

    regularity: FunctionSpaceRegularity
    coefficient_topology: CoefficientTopology


EventInput: TypeAlias = torch.Tensor | npt.ArrayLike


@runtime_checkable
class FunctionSpace(Protocol):
    """Role-neutral construction and evaluation contract."""

    family: FunctionSpaceFamily
    options: Mapping[str, Any]
    metadata: FunctionSpaceMetadata
    feature_count: int

    def evaluate(self, events: EventInput) -> Any:
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

    def build_nuisance_calculation(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "NuisanceCalculation":
        ...


def unexpected_construction_options(
    family: FunctionSpaceFamily,
    construction: Mapping[str, Any],
) -> None:
    """Reject construction arguments that a family did not consume."""

    if construction:
        names = ", ".join(sorted(construction))
        raise TypeError(f"Unexpected {family.value} construction option(s): {names}.")


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

    def build_nuisance_calculation(
        self,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "NuisanceCalculation":
        """Adapt this per-event function space for nuisance evaluation."""

        from neural_networks.nuisance_calculation import NeuralPerEventNuisanceEstimator

        return NeuralPerEventNuisanceEstimator(
            dtype=dtype,
            device=device,
            network=self,
        )


class DeterministicFeatureFunction(PerEventFunctionSpace):
    """Common linear-coefficient topology for fixed feature geometries."""

    metadata = FunctionSpaceMetadata(
        regularity=FunctionSpaceRegularity.DETERMINISTIC,
        coefficient_topology=CoefficientTopology.LINEAR_COEFFICIENTS,
    )

    @classmethod
    def construction_kwargs(
        cls,
        options: Mapping[str, Any],
        construction: dict[str, Any],
    ) -> dict[str, Any]:
        """Consume the construction envelope shared by fixed families."""

        if construction.pop("geometry", None) is not None:
            raise TypeError(f"{cls.family.value} does not accept geometry overrides.")
        result = {
            "dtype": construction.pop("dtype", torch.get_default_dtype()),
            "device": construction.pop("device", None),
            "output_dimension": construction.pop(
                "output_dimension",
                options.get("output_dimension", 1),
            ),
        }
        unexpected_construction_options(cls.family, construction)
        return result

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
        return smoothly_bounded_likelihood_shift(self.evaluate(events))

    def evaluate(self, events: EventInput) -> torch.Tensor:
        return self._linear_evaluation(self.features(events))

    def feature_map(self, events: EventInput) -> torch.Tensor:
        return self.features(events)

    def initialize_parameters(self, gain: float) -> None:
        nn.init.xavier_uniform_(self.coefficients, gain=gain)
