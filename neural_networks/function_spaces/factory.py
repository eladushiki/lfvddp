"""Typed construction of registered function-space families."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

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

    if family_value is FunctionSpaceFamily.ADAPTIVE_NEURAL:
        input_dimension = construction.pop("input_dimension", family_options.get("input_dimension"))
        hidden_size = construction.pop(
            "hidden_size",
            family_options.get("hidden_size", family_options.get("hidden_layer_nodes")),
        )
        output_dimension = construction.pop("output_dimension", family_options.get("output_dimension", 1))
        dtype = construction.pop("dtype", torch.get_default_dtype())
        device = construction.pop("device", None)
        if input_dimension is None or hidden_size is None:
            raise ValueError(
                "adaptive_neural requires input_dimension and hidden_size or hidden_layer_nodes."
            )
        result = registration.factory(
            input_dimension=input_dimension,
            hidden_size=hidden_size,
            output_dimension=output_dimension,
            dtype=dtype,
            device=device,
            options=family_options,
        )
    elif family_value is FunctionSpaceFamily.BIN_INDICATORS:
        # Binned lookup has no tensor parameters, but accepts the common
        # construction envelope used by both role adapters.
        construction.pop("dtype", None)
        construction.pop("device", None)
        construction.pop("output_dimension", None)
        geometry = construction.pop("geometry", None)
        if geometry is None:
            result = registration.factory.from_options(family_options)
        else:
            result = registration.factory(geometry=geometry, options=family_options)
    else:
        dtype = construction.pop("dtype", torch.get_default_dtype())
        device = construction.pop("device", None)
        output_dimension = construction.pop("output_dimension", family_options.get("output_dimension", 1))
        if construction.pop("geometry", None) is not None:
            raise TypeError(f"{family_value.value} does not accept geometry overrides.")
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
