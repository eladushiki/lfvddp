"""Typed construction of registered function-space families."""

from __future__ import annotations

from typing import Any, Mapping, Optional

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

    return registration.factory.from_options(family_options, **construction)


def validate_function_space_options(spec: FunctionSpaceSpec) -> None:
    """Delegate option validation to the selected concrete family."""

    if spec.state is RoleState.DISABLED:
        return
    assert spec.family is not None
    FUNCTION_SPACE_REGISTRY[spec.family].factory.validate_options(spec.options)
