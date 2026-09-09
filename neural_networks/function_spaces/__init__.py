"""Shared function-space contracts, families, registry, and factory."""

from neural_networks.function_spaces.adaptive_neural import (
    AdaptiveNeuralFunction,
    AdaptiveNeuralNetwork,
)
from neural_networks.function_spaces.base import FunctionSpace, FunctionSpaceMetadata
from neural_networks.function_spaces.bin_indicators import (
    BinIndicatorFunction,
    BinIndicatorGeometry,
    BinIndicatorLookup,
)
from neural_networks.function_spaces.factory import create_function_space, function_space_factory
from neural_networks.function_spaces.registry import FUNCTION_SPACE_REGISTRY, FunctionSpaceRegistration

__all__ = [
    "AdaptiveNeuralFunction",
    "AdaptiveNeuralNetwork",
    "BinIndicatorFunction",
    "BinIndicatorGeometry",
    "BinIndicatorLookup",
    "FUNCTION_SPACE_REGISTRY",
    "FunctionSpace",
    "FunctionSpaceMetadata",
    "FunctionSpaceRegistration",
    "create_function_space",
    "function_space_factory",
]
