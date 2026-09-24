"""Factory and configuration coverage shared by every function-space family."""

import pytest
import torch

from neural_networks.function_spaces import (
    AdaptiveNeuralFunction,
    BinIndicatorFunction,
    CubicBSplineFunction,
    FixedSigmoidFunction,
    GaussianRadialBasisFunction,
    OrthogonalPolynomialFunction,
    create_function_space,
)
from train.function_space_config import (
    FunctionSpaceFamily,
    FunctionSpaceRole,
    resolve_dual_role_config,
)
from test.function_space_cases import FUNCTION_SPACE_OPTIONS

FAMILY_TYPES = {
    "adaptive_neural": AdaptiveNeuralFunction,
    "bin_indicators": BinIndicatorFunction,
    "cubic_bspline": CubicBSplineFunction,
    "orthogonal_polynomial": OrthogonalPolynomialFunction,
    "fixed_sigmoid": FixedSigmoidFunction,
    "gaussian_radial_basis": GaussianRadialBasisFunction,
}


def test_all_function_space_families_are_factory_created_for_both_roles():
    assert set(FUNCTION_SPACE_OPTIONS) == {family.value for family in FunctionSpaceFamily}
    for role in (FunctionSpaceRole.F, FunctionSpaceRole.NUISANCE):
        for family, options in FUNCTION_SPACE_OPTIONS.items():
            space = create_function_space(role, family, options, dtype=torch.float64)
            assert type(space) is FAMILY_TYPES[family]
            assert space.evaluate(
                torch.tensor([[0.5], [1.5]], dtype=torch.float64)
            ).shape == (2, 1)


@pytest.mark.parametrize(
    "config, message",
    [
        ({"family": "bin_indicators", "options": {"minima": [0], "maxima": [1]}}, "Missing bin geometry option"),
        ({"family": "fixed_sigmoid", "options": {"centers": [0], "widths": [0]}}, "positive"),
        ({"family": "orthogonal_polynomial", "options": {"basis": "fourier", "maximum_degree": 2, "domain": [0, 1]}}, "legendre"),
        ({"family": "cubic_bspline", "options": {"knots": [0, 1, 1]}}, "knots"),
        ({"family": "gaussian_radial_basis", "options": {"centers": [0]}}, "requires"),
    ],
)
def test_family_owned_option_validation_runs_during_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_dual_role_config(
            f=config,
            nuisance={
                "family": "bin_indicators",
                "options": {"minima": [0], "maxima": [1], "number_of_bins": [2]},
            },
        )


def test_adaptive_neural_construction_requires_derived_dimensions():
    with pytest.raises(ValueError, match="requires input_dimension"):
        create_function_space(
            FunctionSpaceRole.F,
            "adaptive_neural",
            {"input_dimension": 1},
        )
