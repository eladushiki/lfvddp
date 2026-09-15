"""Closed registry of concrete function-space family implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from neural_networks.function_spaces.adaptive_neural import AdaptiveNeuralFunction
from neural_networks.function_spaces.bin_indicators import BinIndicatorFunction
from neural_networks.function_spaces.cubic_bspline import CubicBSplineFunction
from neural_networks.function_spaces.fixed_sigmoid import FixedSigmoidFunction
from neural_networks.function_spaces.gaussian_radial_basis import GaussianRadialBasisFunction
from neural_networks.function_spaces.orthogonal_polynomial import OrthogonalPolynomialFunction
from train.function_space_config import FunctionSpaceFamily


@dataclass(frozen=True)
class FunctionSpaceRegistration:
    """The one factory and metadata definition for a supported family."""

    family: FunctionSpaceFamily
    factory: Callable[..., Any]


FUNCTION_SPACE_REGISTRY: Mapping[FunctionSpaceFamily, FunctionSpaceRegistration] = {
    family: FunctionSpaceRegistration(family, factory)
    for family, factory in (
        (FunctionSpaceFamily.ADAPTIVE_NEURAL, AdaptiveNeuralFunction),
        (FunctionSpaceFamily.BIN_INDICATORS, BinIndicatorFunction),
        (FunctionSpaceFamily.CUBIC_BSPLINE, CubicBSplineFunction),
        (FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL, OrthogonalPolynomialFunction),
        (FunctionSpaceFamily.FIXED_SIGMOID, FixedSigmoidFunction),
        (FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS, GaussianRadialBasisFunction),
    )
}
