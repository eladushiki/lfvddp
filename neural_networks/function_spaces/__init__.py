"""Role-neutral mathematical function-space families and construction."""

from neural_networks.function_spaces.adaptive_neural import AdaptiveNeuralFunction
from neural_networks.function_spaces.base import (
    DeterministicFeatureFunction,
    EventInput,
    FunctionSpace,
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
from neural_networks.function_spaces.factory import (
    analytic_degrees_of_freedom,
    create_function_space,
    prediction_grid_edges,
    validate_function_space_specs,
)
from neural_networks.function_spaces.fixed_sigmoid import FixedSigmoidFunction
from neural_networks.function_spaces.gaussian_radial_basis import GaussianRadialBasisFunction
from neural_networks.function_spaces.orthogonal_polynomial import (
    OrthogonalPolynomialFunction,
    OrthogonalPolynomialGeometry,
    PolynomialBasis,
)

__all__ = [
    "AdaptiveNeuralFunction",
    "analytic_degrees_of_freedom",
    "BinIndicatorFunction",
    "BinIndicatorGeometry",
    "CenterGeometry",
    "CenteredFeatureFunction",
    "CubicBSplineFunction",
    "CubicBSplineGeometry",
    "DeterministicFeatureFunction",
    "EventInput",
    "FixedSigmoidFunction",
    "FunctionSpace",
    "GaussianRadialBasisFunction",
    "OrthogonalPolynomialFunction",
    "OrthogonalPolynomialGeometry",
    "PerEventFunctionSpace",
    "PolynomialBasis",
    "create_function_space",
    "prediction_grid_edges",
    "validate_function_space_specs",
]
