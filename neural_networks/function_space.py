"""Shared mathematical function-space families.

The classes in this module describe reusable mathematical representations.  The
``f`` and ``nuisance`` roles select the same family implementations; likelihood
semantics and training orchestration remain the responsibility of their thin
adapters.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from data_tools.data_utils import DataSet
from neural_networks.likelihood_parameterization import smoothly_bounded_likelihood_shift
from neural_networks.function_spaces.base import FunctionSpaceMetadata
from neural_networks.function_spaces.deterministic import (
    CubicBSplineFunction,
    FixedSigmoidFunction,
    GaussianRadialBasisFunction,
    OrthogonalPolynomialFunction,
)
from train.function_space_config import FunctionSpaceFamily, FunctionSpaceSpec, RoleState


ROLE_LABELS = frozenset({"f", "nuisance"})


def _immutable_options(options: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy factory options so role construction cannot share mutable state."""

    return MappingProxyType(deepcopy(dict(options)))


class AdaptiveNeuralFunction(nn.Module):
    """The existing one-hidden-layer bounded sigmoid network.

    Its module layout and forward calculation intentionally match both legacy
    role-specific network classes so checkpoints and parameter initialization
    remain compatible.
    """

    family = FunctionSpaceFamily.ADAPTIVE_NEURAL
    metadata = FunctionSpaceMetadata(
        regularity="adaptive",
        coefficient_topology="dense_two_layer",
    )

    def __init__(
        self,
        input_dimension: int,
        hidden_size: int,
        output_dimension: int,
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.options = _immutable_options(options or {})
        self.hidden = nn.Linear(
            input_dimension,
            hidden_size,
            dtype=dtype,
            device=device,
        )
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(
            hidden_size,
            output_dimension,
            dtype=dtype,
            device=device,
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return smoothly_bounded_likelihood_shift(
            self.output(self.activation(self.hidden(events)))
        )

    def evaluate(self, events: torch.Tensor) -> torch.Tensor:
        """Evaluate the family without requiring a role-specific adapter."""

        return self.forward(events)


# Descriptive alias for callers that name the implementation as a network.
AdaptiveNeuralNetwork = AdaptiveNeuralFunction


@dataclass(frozen=True)
class BinIndicatorGeometry:
    """Independent detector-bin geometry used by indicator families."""

    minima: tuple[float, ...]
    maxima: tuple[float, ...]
    number_of_bins: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (
            len(self.minima) == len(self.maxima) == len(self.number_of_bins)
        ):
            raise ValueError(
                "Bin geometry minima, maxima, and number_of_bins must have equal lengths."
            )
        if not self.number_of_bins or any(number <= 0 for number in self.number_of_bins):
            raise ValueError("Bin geometry must define a positive number of bins per dimension.")
        if any(minimum >= maximum for minimum, maximum in zip(self.minima, self.maxima)):
            raise ValueError("Each bin geometry minimum must be smaller than its maximum.")

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "BinIndicatorGeometry":
        def dimensions(value: Any, converter: Callable[[Any], Any]) -> tuple[Any, ...]:
            if isinstance(value, (str, bytes)) or np.isscalar(value):
                return (converter(value),)
            return tuple(converter(item) for item in value)

        try:
            return cls(
                minima=dimensions(options["minima"], float),
                maxima=dimensions(options["maxima"], float),
                number_of_bins=dimensions(options["number_of_bins"], int),
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

    def indices(self, events: npt.ArrayLike | DataSet) -> npt.NDArray[np.int64]:
        """Return clipped zero-based indices using DetectorEffect's convention."""

        if isinstance(events, DataSet):
            values = [
                np.asarray(
                    events.slice_along_observable_names(events.observable_names[index])
                ).reshape(-1)
                for index in range(len(self.number_of_bins))
            ]
        else:
            array = np.asarray(events)
            if array.ndim == 1:
                array = array[:, None]
            if array.ndim != 2 or array.shape[1] != len(self.number_of_bins):
                raise ValueError(
                    "Events must have one column per configured bin-geometry dimension."
                )
            values = [array[:, index] for index in range(array.shape[1])]

        indices = []
        for values_for_dimension, edges, number in zip(
            values, self.edges, self.number_of_bins
        ):
            indices.append(
                np.clip(
                    np.digitize(values_for_dimension, edges) - 1,
                    a_min=0,
                    a_max=number - 1,
                )
            )
        return np.column_stack(indices).astype(np.int64, copy=False)


class BinIndicatorFunction:
    """Detector-bin indicator family with a stable evaluation interface."""

    family = FunctionSpaceFamily.BIN_INDICATORS
    metadata = FunctionSpaceMetadata(
        regularity="piecewise_constant",
        coefficient_topology="factorized_marginal_bins",
    )

    def __init__(
        self,
        geometry: BinIndicatorGeometry,
        options: Optional[Mapping[str, Any]] = None,
        detector_effect: Any = None,
    ) -> None:
        self.geometry = geometry
        self.options = _immutable_options(
            options
            if options is not None
            else {
                "minima": list(geometry.minima),
                "maxima": list(geometry.maxima),
                "number_of_bins": list(geometry.number_of_bins),
            }
        )
        self._detector_effect = detector_effect

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "BinIndicatorFunction":
        return cls(BinIndicatorGeometry.from_options(options), options=options)

    @classmethod
    def from_detector_effect(cls, detector_effect: Any) -> "BinIndicatorFunction":
        """Create a lookup from detector geometry without changing detector behavior."""

        names = tuple(detector_effect.observable_names)
        bin_values = [detector_effect.get_observable_bins(name) for name in names]
        geometry = BinIndicatorGeometry(
            minima=tuple(float(edges[0]) for edges, _ in bin_values),
            maxima=tuple(float(edges[-1]) for edges, _ in bin_values),
            number_of_bins=tuple(len(edges) - 1 for edges, _ in bin_values),
        )
        return cls(geometry, detector_effect=detector_effect)

    def bin_indices(self, events: DataSet | npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Evaluate zero-based bin indices for events."""

        if self._detector_effect is not None and isinstance(events, DataSet):
            # This is the legacy nuisance path.  Delegating preserves the exact
            # DetectorEffect edge, clipping, and observable-order semantics.
            return np.asarray(
                self._detector_effect.get_event_bin_centers(events, indexed=True),
                dtype=np.int64,
            )
        return self.geometry.indices(events)

    def evaluate(self, events: DataSet | npt.ArrayLike) -> npt.NDArray[np.int64]:
        return self.bin_indices(events)


# The lookup name is useful to adapters that consume indices rather than a
# feature matrix; both names intentionally refer to one implementation.
BinIndicatorLookup = BinIndicatorFunction


def _copy_family_options(options: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(options))


def initialize_function_space_parameters(function_space: nn.Module, gain: float) -> None:
    """Initialize any trainable coefficients using the common project policy."""
    if hasattr(function_space, "hidden") and hasattr(function_space, "output"):
        nn.init.xavier_uniform_(function_space.hidden.weight, gain=gain)
        nn.init.uniform_(function_space.hidden.bias, a=-0.3, b=0.3)
        nn.init.xavier_uniform_(function_space.output.weight, gain=gain)
        nn.init.uniform_(function_space.output.bias, a=-0.3, b=0.3)
        return
    coefficients = getattr(function_space, "coefficients", None)
    if isinstance(coefficients, nn.Parameter):
        nn.init.xavier_uniform_(coefficients, gain=gain)
        return
    raise TypeError(
        f"Function-space {type(function_space).__name__} has no supported trainable parameters."
    )


@dataclass(frozen=True)
class FunctionSpaceRegistration:
    family: FunctionSpaceFamily
    factory: Callable[..., Any]
    option_parser: Optional[Callable[[Mapping[str, Any]], Mapping[str, Any]]] = None
    metadata: Optional[FunctionSpaceMetadata] = None


FUNCTION_SPACE_REGISTRY: Mapping[FunctionSpaceFamily, FunctionSpaceRegistration] = {
    FunctionSpaceFamily.ADAPTIVE_NEURAL: FunctionSpaceRegistration(
        FunctionSpaceFamily.ADAPTIVE_NEURAL,
        AdaptiveNeuralFunction,
        option_parser=_copy_family_options,
        metadata=AdaptiveNeuralFunction.metadata,
    ),
    FunctionSpaceFamily.BIN_INDICATORS: FunctionSpaceRegistration(
        FunctionSpaceFamily.BIN_INDICATORS,
        BinIndicatorFunction,
        option_parser=_copy_family_options,
        metadata=BinIndicatorFunction.metadata,
    ),
    FunctionSpaceFamily.CUBIC_BSPLINE: FunctionSpaceRegistration(
        FunctionSpaceFamily.CUBIC_BSPLINE,
        CubicBSplineFunction,
        option_parser=_copy_family_options,
        metadata=CubicBSplineFunction.metadata,
    ),
    FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL: FunctionSpaceRegistration(
        FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL,
        OrthogonalPolynomialFunction,
        option_parser=_copy_family_options,
        metadata=OrthogonalPolynomialFunction.metadata,
    ),
    FunctionSpaceFamily.FIXED_SIGMOID: FunctionSpaceRegistration(
        FunctionSpaceFamily.FIXED_SIGMOID,
        FixedSigmoidFunction,
        option_parser=_copy_family_options,
        metadata=FixedSigmoidFunction.metadata,
    ),
    FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS: FunctionSpaceRegistration(
        FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS,
        GaussianRadialBasisFunction,
        option_parser=_copy_family_options,
        metadata=GaussianRadialBasisFunction.metadata,
    ),
}


def _family_from_spec(
    family: FunctionSpaceFamily | str | FunctionSpaceSpec,
    options: Optional[Mapping[str, Any]],
) -> tuple[FunctionSpaceFamily, Mapping[str, Any], RoleState]:
    if isinstance(family, FunctionSpaceSpec):
        if options is not None:
            raise ValueError("Options cannot be supplied twice for a FunctionSpaceSpec.")
        return family.family, family.options, family.state  # type: ignore[return-value]
    return FunctionSpaceFamily.from_value(family), options or {}, RoleState.ENABLED


def create_function_space(
    role: str,
    family: FunctionSpaceFamily | str | FunctionSpaceSpec,
    options: Optional[Mapping[str, Any]] = None,
    **construction: Any,
) -> Any:
    """Construct one supported family for either the ``f`` or ``nuisance`` role."""

    if role not in ROLE_LABELS:
        raise ValueError(f"Unknown function-space role {role!r}; expected 'f' or 'nuisance'.")
    family_value, family_options, state = _family_from_spec(family, options)
    if state is RoleState.DISABLED:
        raise ValueError(f"Cannot construct a disabled {role} function-space role.")
    try:
        registration = FUNCTION_SPACE_REGISTRY[family_value]
    except KeyError as error:
        supported = ", ".join(item.value for item in FUNCTION_SPACE_REGISTRY)
        raise ValueError(
            f"Function-space family {family_value.value!r} is not implemented; "
            f"supported families: {supported}."
        ) from error

    copied_options = (
        registration.option_parser(family_options)
        if registration.option_parser is not None
        else deepcopy(dict(family_options))
    )
    if family_value is FunctionSpaceFamily.ADAPTIVE_NEURAL:
        input_dimension = construction.pop(
            "input_dimension", copied_options.pop("input_dimension", None)
        )
        hidden_size = construction.pop(
            "hidden_size",
            copied_options.pop("hidden_size", copied_options.pop("hidden_layer_nodes", None)),
        )
        output_dimension = construction.pop(
            "output_dimension", copied_options.pop("output_dimension", 1)
        )
        dtype = construction.pop("dtype", torch.get_default_dtype())
        device = construction.pop("device", None)
        if input_dimension is None or hidden_size is None:
            raise ValueError(
                "adaptive_neural requires input_dimension and hidden_size (or hidden_layer_nodes)."
            )
        result = registration.factory(
            input_dimension=input_dimension,
            hidden_size=hidden_size,
            output_dimension=output_dimension,
            dtype=dtype,
            device=device,
            options=family_options,
        )
    else:
        dtype = construction.pop("dtype", torch.get_default_dtype())
        device = construction.pop("device", None)
        output_dimension = construction.pop(
            "output_dimension", copied_options.pop("output_dimension", 1)
        )
        detector_effect = construction.pop("detector_effect", None)
        geometry = construction.pop("geometry", None)
        if family_value is FunctionSpaceFamily.BIN_INDICATORS:
            if geometry is None:
                result = (
                    registration.factory.from_detector_effect(detector_effect)
                    if detector_effect is not None
                    else registration.factory.from_options(family_options)
                )
            else:
                result = registration.factory(
                    geometry=geometry, options=family_options, detector_effect=detector_effect
                )
        else:
            if detector_effect is not None or geometry is not None:
                raise TypeError(
                    f"{family_value.value} does not accept detector_effect or geometry overrides."
                )
            result = registration.factory.from_options(
                family_options,
                dtype=dtype,
                device=device,
                output_dimension=output_dimension,
            )
    if construction:
        names = ", ".join(sorted(construction))
        raise TypeError(f"Unexpected {family_value.value} construction option(s): {names}.")
    return result


# Short alias for adapters that prefer factory terminology.
function_space_factory = create_function_space


__all__ = [
    "AdaptiveNeuralFunction",
    "AdaptiveNeuralNetwork",
    "BinIndicatorFunction",
    "BinIndicatorGeometry",
    "BinIndicatorLookup",
    "FUNCTION_SPACE_REGISTRY",
    "FunctionSpaceRegistration",
    "create_function_space",
    "function_space_factory",
]
