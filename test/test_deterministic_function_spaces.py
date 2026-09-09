import pytest
import torch

from neural_networks.function_spaces import (
    CubicBSplineFunction,
    FixedSigmoidFunction,
    GaussianRadialBasisFunction,
    OrthogonalPolynomialFunction,
    create_function_space,
)
from train.function_space_config import resolve_dual_role_config


FAMILY_OPTIONS = {
    "cubic_bspline": {"knots": [0.0, 1.0, 2.0, 3.0]},
    "orthogonal_polynomial": {
        "basis": "legendre",
        "maximum_degree": 2,
        "domain": [0.0, 3.0],
    },
    "fixed_sigmoid": {"centers": [-1.0, 1.0], "widths": [0.5, 0.5]},
    "gaussian_radial_basis": {"centers": [-1.0, 1.0], "widths": [0.5, 0.5]},
}


def test_all_deterministic_families_are_factory_created_for_both_roles():
    expected = {
        "cubic_bspline": CubicBSplineFunction,
        "orthogonal_polynomial": OrthogonalPolynomialFunction,
        "fixed_sigmoid": FixedSigmoidFunction,
        "gaussian_radial_basis": GaussianRadialBasisFunction,
    }
    for role in ("f", "nuisance"):
        for family, options in FAMILY_OPTIONS.items():
            space = create_function_space(role, family, options, dtype=torch.float64)
            assert type(space) is expected[family]
            assert space.features(torch.tensor([[0.5], [1.5]], dtype=torch.float64)).shape == (
                2,
                space.feature_count,
            )
            assert space.evaluate(torch.tensor([[0.5], [1.5]], dtype=torch.float64)).shape == (2, 1)


def test_feature_counts_and_multidimensional_geometry_are_explicit():
    spline = create_function_space(
        "f", "cubic_bspline", {"knots": [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]}
    )
    polynomial = create_function_space(
        "f",
        "orthogonal_polynomial",
        {"basis": "legendre", "maximum_degree": 2, "domain": [[0, 1], [0, 2]]},
    )
    radial = create_function_space(
        "f",
        "gaussian_radial_basis",
        {"centers": [[0, 0], [1, 2]], "widths": [[1, 2], [2, 1]]},
    )
    assert spline.input_dimension == 2 and spline.feature_count == 10
    assert polynomial.input_dimension == 2 and polynomial.feature_count == 6
    assert radial.input_dimension == 2 and radial.feature_count == 2
    assert spline.features(torch.tensor([[0.5, 1.5]])).shape == (1, 10)
    assert radial.features(torch.tensor([[0.0, 0.0]])).shape == (1, 2)


def test_spline_partition_of_unity_and_fixed_geometry_boundary_behavior():
    spline = create_function_space("f", "cubic_bspline", {"knots": [0.0, 1.0, 2.0, 3.0]})
    values = spline.features(torch.tensor([[0.0], [1.0], [3.0], [-1.0], [4.0]]))
    assert torch.allclose(values[:3].sum(dim=1), torch.ones(3))
    assert torch.equal(values[3:], torch.zeros((2, spline.feature_count)))
    assert all(not parameter.requires_grad for name, parameter in spline.named_buffers())
    assert [name for name, _ in spline.named_parameters()] == ["coefficients"]


def test_polynomial_basis_is_normalized_and_legendre_differs_from_chebyshev():
    options = {"maximum_degree": 2, "domain": [0.0, 2.0]}
    legendre = create_function_space("f", "orthogonal_polynomial", {**options, "basis": "legendre"})
    chebyshev = create_function_space("f", "orthogonal_polynomial", {**options, "basis": "chebyshev"})
    at_midpoint = torch.tensor([[1.0]])
    assert torch.allclose(legendre.features(at_midpoint), torch.tensor([[1.0, 0.0, -0.5]]))
    assert torch.allclose(chebyshev.features(at_midpoint), torch.tensor([[1.0, 0.0, -1.0]]))
    assert not torch.equal(legendre.features(torch.tensor([[0.25]])), chebyshev.features(torch.tensor([[0.25]])))


def test_fixed_sigmoid_and_radial_values_use_their_documented_formulas():
    sigmoid = create_function_space(
        "f", "fixed_sigmoid", {"centers": [0.0], "widths": [1.0]}
    )
    radial = create_function_space(
        "nuisance", "gaussian_radial_basis", {"centers": [0.0], "widths": [1.0]}
    )
    events = torch.tensor([[0.0], [1.0]])
    assert torch.allclose(sigmoid.features(events), torch.tensor([[0.5], [torch.sigmoid(torch.tensor(1.0))]]))
    assert torch.allclose(radial.features(events), torch.tensor([[1.0], [torch.exp(torch.tensor(-0.5))]]))


def test_fixed_maps_are_linear_in_their_common_coefficients():
    for family, options in FAMILY_OPTIONS.items():
        space = create_function_space("nuisance", family, options, dtype=torch.float64)
        with torch.no_grad():
            space.coefficients.copy_(torch.arange(space.feature_count, dtype=torch.float64)[:, None])
        events = torch.tensor([[-0.25], [0.75]], dtype=torch.float64)
        features = space.features(events)
        assert torch.allclose(space.evaluate(events), features @ space.coefficients)
        baseline = space.evaluate(events).detach()
        with torch.no_grad():
            space.coefficients.mul_(2.0)
        assert torch.allclose(space.evaluate(events), 2.0 * baseline)


def test_geometry_options_are_copied_and_not_trainable():
    options = {"centers": [[0.0, 1.0], [2.0, 3.0]], "widths": [[1.0, 1.0], [1.0, 1.0]]}
    space = create_function_space("f", "gaussian_radial_basis", options)
    options["centers"][0][0] = 99.0
    options["widths"][0][0] = 99.0
    assert space.geometry.centers[0][0] == 0.0
    assert space.geometry.widths[0][0] == 1.0
    with pytest.raises(TypeError):
        space.options["centers"] = ()
    assert all(not parameter.requires_grad for parameter in space.buffers())


def test_dtype_and_device_follow_the_constructed_module():
    space = create_function_space(
        "nuisance", "fixed_sigmoid", FAMILY_OPTIONS["fixed_sigmoid"], dtype=torch.float64
    )
    output = space.evaluate(torch.tensor([[0.0]], dtype=torch.float32))
    assert output.dtype is torch.float64 and output.device == space.coefficients.device
    if torch.cuda.is_available():
        space = space.to("cuda")
        output = space.evaluate(torch.tensor([[0.0]], device="cuda"))
        assert output.device.type == "cuda" and output.dtype is torch.float64


@pytest.mark.parametrize(
    "config, message",
    [
        ({"family": "fixed_sigmoid", "options": {"centers": [0], "widths": [0]}}, "positive"),
        ({"family": "orthogonal_polynomial", "options": {"basis": "fourier", "maximum_degree": 2, "domain": [0, 1]}}, "legendre"),
        ({"family": "cubic_bspline", "options": {"knots": [0, 1, 1]}}, "knots"),
        ({"family": "gaussian_radial_basis", "options": {"centers": [0]}}, "requires"),
    ],
)
def test_invalid_deterministic_options_fail_canonically(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_dual_role_config(f=config, nuisance={"family": "bin_indicators", "options": {"minima": [0], "maxima": [1], "number_of_bins": [2]}})
