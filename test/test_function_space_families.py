import numpy as np
import pytest
import torch

from neural_networks.function_spaces import (
    AdaptiveNeuralFunction,
    BinIndicatorFunction,
    CoefficientTopology,
    FUNCTION_SPACE_REGISTRY,
    FunctionSpaceRegularity,
    create_function_space,
)
from neural_networks.nuisance_calculation import (
    PerEventNuisanceEstimator,
    BinnedNuisanceCalculation,
)
from train.function_space_config import FunctionSpaceFamily, FunctionSpaceRole


def test_registry_constructs_supported_families_for_both_typed_roles():
    assert set(FUNCTION_SPACE_REGISTRY) == set(FunctionSpaceFamily)

    for role in FunctionSpaceRole:
        adaptive = create_function_space(
            role,
            FunctionSpaceFamily.ADAPTIVE_NEURAL,
            {"input_dimension": 2, "hidden_layer_nodes": 3},
            dtype=torch.float64,
        )
        binned = create_function_space(
            role,
            FunctionSpaceFamily.BIN_INDICATORS,
            {"minima": [0.0], "maxima": [10.0], "number_of_bins": [4]},
        )
        assert isinstance(adaptive, AdaptiveNeuralFunction)
        assert isinstance(binned, BinIndicatorFunction)
        assert adaptive.prediction_grid_edges() is None
        np.testing.assert_equal(binned.prediction_grid_edges(), binned.geometry.edges)


def test_role_construction_does_not_alias_geometry_or_options():
    options = {
        "minima": [0.0],
        "maxima": [10.0],
        "number_of_bins": [4],
    }
    f_space = create_function_space(
        FunctionSpaceRole.F,
        FunctionSpaceFamily.BIN_INDICATORS,
        options,
    )
    nuisance_space = create_function_space(
        FunctionSpaceRole.NUISANCE,
        FunctionSpaceFamily.BIN_INDICATORS,
        options,
    )

    options["minima"][0] = 100.0
    assert f_space.geometry.minima == (0.0,)
    assert nuisance_space.geometry.minima == (0.0,)
    assert f_space.geometry is not nuisance_space.geometry
    assert f_space.options is not nuisance_space.options


def test_bin_indices_depend_only_on_function_space_geometry():
    lookup = BinIndicatorFunction.from_options(
        {"minima": [0.0, -1.0], "maxima": [2.0, 1.0], "number_of_bins": [2, 2]}
    )
    events = np.array([[0.1, -0.8], [1.9, 0.8]])

    np.testing.assert_array_equal(lookup.evaluate(events), [[0, 0], [1, 1]])


def test_metadata_categories_are_enums_for_every_registered_family():
    for registration in FUNCTION_SPACE_REGISTRY.values():
        metadata = registration.factory.metadata
        assert isinstance(metadata.regularity, FunctionSpaceRegularity)
        assert isinstance(metadata.coefficient_topology, CoefficientTopology)


def test_family_owned_nuisance_adapters_retain_their_function_space_instances():
    adaptive = create_function_space(
        FunctionSpaceRole.NUISANCE,
        FunctionSpaceFamily.ADAPTIVE_NEURAL,
        {"input_dimension": 2, "hidden_layer_nodes": 3},
        dtype=torch.float64,
    )
    binned = create_function_space(
        FunctionSpaceRole.NUISANCE,
        FunctionSpaceFamily.BIN_INDICATORS,
        {"minima": [0.0], "maxima": [10.0], "number_of_bins": [4]},
    )

    neural_adapter = adaptive.build_nuisance_calculation(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    binned_adapter = binned.build_nuisance_calculation(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert isinstance(neural_adapter, PerEventNuisanceEstimator)
    assert neural_adapter.network is adaptive
    assert isinstance(binned_adapter, BinnedNuisanceCalculation)
    assert binned_adapter.function_space is binned


def test_shared_neural_family_preserves_role_adapter_shapes():
    torch.manual_seed(7)
    signal = AdaptiveNeuralFunction(2, 3, 1, torch.float64)
    torch.manual_seed(7)
    nuisance = AdaptiveNeuralFunction(2, 3, 1, torch.float64)
    nuisance_adapter = nuisance.build_nuisance_calculation(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert type(signal) is type(nuisance) is AdaptiveNeuralFunction
    assert signal is not nuisance
    assert nuisance_adapter.network is nuisance
    assert list(signal.state_dict()) == list(nuisance.state_dict())
    nuisance.load_state_dict(signal.state_dict())
    events = torch.ones(5, 2, dtype=torch.float64)
    assert signal(events).shape == (5, 1)
    assert nuisance(events).shape == (5, 1)
    torch.testing.assert_close(signal(events), nuisance(events))
