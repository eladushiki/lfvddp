"""Closed registry of currently implemented role-neutral families."""

from neural_networks.function_space import (
    FUNCTION_SPACE_REGISTRY,
    FunctionSpaceRegistration,
)

__all__ = ["FUNCTION_SPACE_REGISTRY", "FunctionSpaceRegistration"]
