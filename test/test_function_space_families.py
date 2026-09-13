import numpy as np
import pytest
import torch

from data_tools.data_utils import DataSet
from neural_networks.differentiating_model import _SignalRegionShiftEstimator
from neural_networks.function_spaces import (
    AdaptiveNeuralFunction,
    BinIndicatorFunction,
    FUNCTION_SPACE_REGISTRY,
    create_function_space,
)
from neural_networks.nuisance_calculation import _ThetaEstimator

def test_registry_constructs_supported_families_for_both_roles():
    assert {family.value for family in FUNCTION_SPACE_REGISTRY} == {
        "adaptive_neural",
        "bin_indicators",
        "cubic_bspline",
        "orthogonal_polynomial",
        "fixed_sigmoid",
        "gaussian_radial_basis",
    }

    for role in ("f", "nuisance"):
        adaptive = create_function_space(
            role,
            "adaptive_neural",
            {"input_dimension": 2, "hidden_layer_nodes": 3},
            dtype=torch.float64,
        )
        binned = create_function_space(
            role,
            "bin_indicators",
            {"minima": [0.0], "maxima": [10.0], "number_of_bins": [4]},
        )
        assert isinstance(adaptive, AdaptiveNeuralFunction)
        assert isinstance(binned, BinIndicatorFunction)


def test_role_construction_does_not_alias_geometry_or_options():
    options = {
        "minima": [0.0],
        "maxima": [10.0],
        "number_of_bins": [4],
    }
    f_space = create_function_space("f", "bin_indicators", options)
    nuisance_space = create_function_space("nuisance", "bin_indicators", options)

    options["minima"][0] = 100.0
    assert f_space.geometry.minima == (0.0,)
    assert nuisance_space.geometry.minima == (0.0,)
    assert f_space.geometry is not nuisance_space.geometry
    assert f_space.options is not nuisance_space.options


def test_bin_indices_follow_canonical_function_space_geometry():
    lookup = BinIndicatorFunction.from_options(
        {"minima": [0.0, -1.0], "maxima": [2.0, 1.0], "number_of_bins": [2, 2]}
    )
    events = DataSet(
        np.array([[0.1, -0.8], [1.9, 0.8]]),
        observable_names=["x", "y"],
    )
    np.testing.assert_array_equal(lookup.evaluate(events), [[0, 0], [1, 1]])
def test_role_specific_network_names_preserve_shared_structure():
    assert _SignalRegionShiftEstimator is AdaptiveNeuralFunction

    torch.manual_seed(7)
    signal = _SignalRegionShiftEstimator(2, 3, 1, torch.float64)
    torch.manual_seed(7)
    nuisance = _ThetaEstimator(2, 3, 1, torch.float64)

    assert list(signal.state_dict()) == list(nuisance.state_dict())
    nuisance.load_state_dict(signal.state_dict())
    events = torch.ones(5, 2, dtype=torch.float64)
    assert signal(events).shape == (5, 1)
    assert nuisance(events).shape == (5,)
    torch.testing.assert_close(signal(events).squeeze(-1), nuisance(events))
