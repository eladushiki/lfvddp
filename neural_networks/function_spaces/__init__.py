"""Shared function-space contracts, families, registry, and factory."""

from neural_networks.function_spaces.adaptive_neural import (
    AdaptiveNeuralFunction,
    AdaptiveNeuralNetwork,
    initialize_function_space_parameters,
)
from neural_networks.function_spaces.base import FunctionSpace, FunctionSpaceMetadata
from neural_networks.function_spaces.bin_indicators import (
    BinIndicatorFunction,
    BinIndicatorGeometry,
    BinIndicatorLookup,
)
from neural_networks.function_spaces.projected_rank import (
    ProjectedFunctionSpaceRank,
    compute_projected_function_space_rank,
    compute_projected_rank,
    compute_rank_for_backend,
    projected_function_space_rank,
)
from neural_networks.function_spaces.deterministic import (
    CenterGeometry,
    CubicBSpline,
    CubicBSplineFunction,
    CubicBSplineGeometry,
    DeterministicFeatureFunction,
    FixedSigmoid,
    FixedSigmoidFunction,
    GaussianRadialBasis,
    GaussianRadialBasisFunction,
    OrthogonalPolynomial,
    OrthogonalPolynomialFunction,
    OrthogonalPolynomialGeometry,
)
from neural_networks.function_spaces.factory import create_function_space, function_space_factory
from neural_networks.function_spaces.registry import FUNCTION_SPACE_REGISTRY, FunctionSpaceRegistration

__all__ = [
    "AdaptiveNeuralFunction",
    "AdaptiveNeuralNetwork",
    "initialize_function_space_parameters",
    "BinIndicatorFunction",
    "BinIndicatorGeometry",
    "BinIndicatorLookup",
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
    "FUNCTION_SPACE_REGISTRY",
    "FunctionSpace",
    "FunctionSpaceMetadata",
    "FunctionSpaceRegistration",
    "create_function_space",
    "function_space_factory",
    "ProjectedFunctionSpaceRank",
    "compute_projected_function_space_rank",
    "compute_projected_rank",
    "compute_rank_for_backend",
    "projected_function_space_rank",
]
