"""Construction and validation for the registered likelihood function spaces."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from data_tools.data_utils import ShiftAndNormalizationFactor
from neural_networks.function_spaces.bin_indicators import BinIndicatorGeometry
from neural_networks.function_spaces.registry import FUNCTION_SPACE_REGISTRY
from train.function_space_config import FunctionSpaceSpec


def _registered_family(spec: FunctionSpaceSpec):
    try:
        return FUNCTION_SPACE_REGISTRY[spec.family]
    except KeyError as error:
        supported = ", ".join(sorted(FUNCTION_SPACE_REGISTRY))
        raise ValueError(
            f"Function-space family {spec.family!r} is not implemented; "
            f"supported families: {supported}."
        ) from error


def validate_function_space_options(spec: FunctionSpaceSpec) -> None:
    """Delegate immutable option validation to the selected family."""

    _registered_family(spec).validate_options(spec.options)


def validate_function_space_specs(
    f: FunctionSpaceSpec,
    nuisance: FunctionSpaceSpec | None,
) -> None:
    """Validate every enabled canonical function-space specification."""

    validate_function_space_options(f)
    if nuisance is not None:
        validate_function_space_options(nuisance)


def create_function_space(
    spec: FunctionSpaceSpec,
    *,
    dtype,
    device=None,
    normalization_factor: Optional[ShiftAndNormalizationFactor] = None,
    observable_names: Optional[Iterable[str]] = None,
    **construction: Any,
):
    """Build one normalized-event likelihood shift from its canonical spec."""

    if (normalization_factor is None) != (observable_names is None):
        raise ValueError(
            "Function-space construction requires both normalization and observables."
        )
    validate_function_space_options(spec)
    function_space = _registered_family(spec).from_options(
        spec.options,
        dtype=dtype,
        device=device,
        **construction,
    )
    if normalization_factor is not None:
        function_space.normalize_input_geometry(
            normalization_factor, tuple(observable_names)
        )
    return function_space


def prediction_grid_edges(
    spec: FunctionSpaceSpec | None,
) -> tuple | None:
    """Return immutable physical bin edges for plotting when a spec is binned."""

    if spec is None or spec.family != "bin_indicators":
        return None
    return BinIndicatorGeometry.from_options(spec.options).edges


def analytic_degrees_of_freedom(spec: FunctionSpaceSpec) -> int | None:
    """Return a family-owned fixed dimension without allocating tensors."""

    return _registered_family(spec).analytic_degrees_of_freedom(spec.options)
