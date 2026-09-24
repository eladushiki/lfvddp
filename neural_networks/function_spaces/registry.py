"""Closed registry of concrete function-space family implementations."""

from __future__ import annotations

from typing import Mapping, Type

from neural_networks.function_spaces.adaptive_neural import AdaptiveNeuralFunction
from neural_networks.function_spaces.bin_indicators import BinIndicatorFunction
from neural_networks.function_spaces.cubic_bspline import CubicBSplineFunction
from neural_networks.function_spaces.fixed_sigmoid import FixedSigmoidFunction
from neural_networks.function_spaces.gaussian_radial_basis import GaussianRadialBasisFunction
from neural_networks.function_spaces.orthogonal_polynomial import OrthogonalPolynomialFunction
FunctionSpaceType = Type[
    AdaptiveNeuralFunction
    | BinIndicatorFunction
    | CubicBSplineFunction
    | FixedSigmoidFunction
    | GaussianRadialBasisFunction
    | OrthogonalPolynomialFunction
]


FUNCTION_SPACE_REGISTRY: Mapping[str, FunctionSpaceType] = {
    function_space.family: function_space
    for function_space in (
        AdaptiveNeuralFunction,
        BinIndicatorFunction,
        CubicBSplineFunction,
        OrthogonalPolynomialFunction,
        FixedSigmoidFunction,
        GaussianRadialBasisFunction,
    )
}
