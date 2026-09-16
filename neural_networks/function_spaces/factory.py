"""Typed construction of registered function-space families."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from data_tools.data_utils import ShiftAndNormalizationFactor

from neural_networks.function_spaces.registry import FUNCTION_SPACE_REGISTRY
from train.function_space_config import (
    FunctionSpaceFamily,
    FunctionSpaceRole,
    FunctionSpaceSpec,
    RoleState,
)


def _family_and_options(
    family: FunctionSpaceFamily | str | FunctionSpaceSpec,
    options: Optional[Mapping[str, Any]],
) -> tuple[FunctionSpaceFamily, Mapping[str, Any], RoleState]:
    if isinstance(family, FunctionSpaceSpec):
        if options is not None:
            raise ValueError("Options cannot be supplied twice for a FunctionSpaceSpec.")
        if family.family is None:
            return FunctionSpaceFamily.ADAPTIVE_NEURAL, family.options, family.state
        return family.family, family.options, family.state
    return FunctionSpaceFamily.from_value(family), options or {}, RoleState.ENABLED


def create_function_space(
    role: FunctionSpaceRole | str,
    family: FunctionSpaceFamily | str | FunctionSpaceSpec,
    options: Optional[Mapping[str, Any]] = None,
    **construction: Any,
) -> Any:
    """Construct one registered family for an explicitly typed likelihood role."""

    role_value = FunctionSpaceRole.from_value(role)
    normalization_factor = construction.pop("normalization_factor", None)
    observable_names = construction.pop("observable_names", None)
    if (normalization_factor is None) != (observable_names is None):
        raise ValueError(
            "Function-space construction requires both normalization and observables."
        )
    if normalization_factor is not None and not isinstance(
        normalization_factor, ShiftAndNormalizationFactor
    ):
        raise TypeError("normalization_factor must be a ShiftAndNormalizationFactor.")
    if observable_names is not None:
        observable_names = tuple(observable_names)
    family_value, family_options, state = _family_and_options(family, options)
    if state is RoleState.DISABLED:
        raise ValueError(f"Cannot construct a disabled {role_value.value} function-space role.")
    try:
        registration = FUNCTION_SPACE_REGISTRY[family_value]
    except KeyError as error:
        supported = ", ".join(item.value for item in FUNCTION_SPACE_REGISTRY)
        raise ValueError(
            f"Function-space family {family_value.value!r} is not implemented; "
            f"supported families: {supported}."
        ) from error

    function_space = registration.factory.from_options(family_options, **construction)
    if normalization_factor is not None:
        function_space.normalize_input_geometry(normalization_factor, observable_names)
    return function_space


def validate_function_space_options(spec: FunctionSpaceSpec) -> None:
    """Delegate option validation to the selected concrete family."""

    if spec.state is RoleState.DISABLED:
        return
    assert spec.family is not None
    FUNCTION_SPACE_REGISTRY[spec.family].factory.validate_options(spec.options)
