"""Role-neutral mathematical function-space families and typed construction."""

from neural_networks.function_spaces.adaptive_neural import AdaptiveNeuralFunction
from neural_networks.function_spaces.base import (
    CoefficientTopology,
    DeterministicFeatureFunction,
    EventInput,
    FunctionSpace,
    FunctionSpaceMetadata,
    FunctionSpaceRegularity,
    PerEventFunctionSpace,
)
from neural_networks.function_spaces.bin_indicators import (
    BinIndicatorFunction,
    BinIndicatorGeometry,
)
from neural_networks.function_spaces.centered import CenterGeometry, CenteredFeatureFunction
from neural_networks.function_spaces.cubic_bspline import (
    CubicBSplineFunction,
    CubicBSplineGeometry,
)
from neural_networks.function_spaces.factory import create_function_space
from neural_networks.function_spaces.fixed_sigmoid import FixedSigmoidFunction
from neural_networks.function_spaces.gaussian_radial_basis import GaussianRadialBasisFunction
from neural_networks.function_spaces.orthogonal_polynomial import (
    OrthogonalPolynomialFunction,
    OrthogonalPolynomialGeometry,
    PolynomialBasis,
)
from neural_networks.function_spaces.registry import FUNCTION_SPACE_REGISTRY, FunctionSpaceRegistration

__all__ = [
    "AdaptiveNeuralFunction",
    "BinIndicatorFunction",
    "BinIndicatorGeometry",
    "CenterGeometry",
    "CenteredFeatureFunction",
    "CubicBSplineFunction",
    "CubicBSplineGeometry",
    "DeterministicFeatureFunction",
    "EventInput",
    "FixedSigmoidFunction",
    "FUNCTION_SPACE_REGISTRY",
    "FunctionSpace",
    "FunctionSpaceMetadata",
    "FunctionSpaceRegistration",
    "GaussianRadialBasisFunction",
    "OrthogonalPolynomialFunction",
    "OrthogonalPolynomialGeometry",
    "PolynomialBasis",
    "create_function_space",
]
