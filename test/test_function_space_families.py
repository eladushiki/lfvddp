import torch

from neural_networks.function_spaces import BinIndicatorFunction, create_function_space
from train.function_space_config import FunctionSpaceSpec


def test_independent_role_specs_do_not_alias_geometry_or_options():
    options = {"minima": [0.0], "maxima": [10.0], "number_of_bins": [4]}
    f_space = create_function_space(FunctionSpaceSpec("bin_indicators", options), dtype=torch.float64)
    nuisance_space = create_function_space(
        FunctionSpaceSpec("bin_indicators", options), dtype=torch.float64
    )

    options["minima"][0] = 100.0
    assert f_space.geometry.minima == (0.0,)
    assert nuisance_space.geometry.minima == (0.0,)
    assert f_space.geometry is not nuisance_space.geometry
    assert f_space.options is not nuisance_space.options


def test_binned_space_has_one_independent_coefficient_per_cartesian_cell():
    space = create_function_space(
        FunctionSpaceSpec(
            "bin_indicators",
            {"minima": [0.0, -1.0], "maxima": [2.0, 1.0], "number_of_bins": [2, 2]},
        ),
        dtype=torch.float64,
    )
    assert isinstance(space, BinIndicatorFunction)
    assert space.feature_count == 4
    assert space.coefficients.shape == (4, 1)
    features = space.features(torch.tensor([[0.1, -0.8], [1.9, 0.8]], dtype=torch.float64))
    assert torch.equal(features.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert not torch.equal(features[0], features[1])
