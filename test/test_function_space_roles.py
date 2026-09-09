from types import SimpleNamespace

import pytest
import torch

from neural_networks.differentiating_model import DifferentiatingModel
from neural_networks.function_spaces import AdaptiveNeuralFunction, BinIndicatorFunction
from neural_networks.nuisance_calculation import (
    BlankNuisanceEstimator,
    NeuralPerEventNuisanceEstimator,
    ScalarBinnedNuisanceEstimator,
    build_nuisance_calculation,
)
from train.function_space_config import FunctionSpaceFamily, resolve_dual_role_config
from train.train_config import TrainConfig


class _Detector:
    observable_names = ("x", "y")


def _context(*, f_options, nuisance):
    config = TrainConfig(
        train__epochs=1,
        train__number_of_epochs_for_checkpoint=1,
        train__nn_inner_layer_nodes=2,
        train__nn_input_dimension=2,
        train__function_space={
            "f": {"family": "adaptive_neural", "options": f_options},
            "nuisance": nuisance,
        },
    )
    return SimpleNamespace(config=config)


def test_model_builds_independent_same_family_role_adapters():
    context = _context(
        f_options={"input_dimension": 2, "hidden_layer_nodes": 5},
        nuisance={
            "family": "adaptive_neural",
            "options": {"input_dimension": 2, "hidden_layer_nodes": 3},
        },
    )

    model = DifferentiatingModel(
        context=context,
        detector_effect=_Detector(),
        is_numerator=True,
        name="canonical_roles",
        dtype=torch.float64,
    )

    assert isinstance(model.signal_region_shift_network, AdaptiveNeuralFunction)
    assert isinstance(model.nuisance_calculation, NeuralPerEventNuisanceEstimator)
    assert isinstance(model.nuisance_calculation.network, AdaptiveNeuralFunction)
    assert model.signal_region_shift_network.hidden.out_features == 5
    assert model.nuisance_calculation.network.hidden.out_features == 3
    assert model.signal_region_shift_network is not model.nuisance_calculation.network
    assert tuple(model.state_dict()) == (
        "nuisance_calculation.network.hidden.weight",
        "nuisance_calculation.network.hidden.bias",
        "nuisance_calculation.network.output.weight",
        "nuisance_calculation.network.output.bias",
        "signal_region_shift_network.hidden.weight",
        "signal_region_shift_network.hidden.bias",
        "signal_region_shift_network.output.weight",
        "signal_region_shift_network.output.bias",
    )


def test_f_remains_enabled_when_nuisance_is_disabled():
    context = _context(
        f_options={"input_dimension": 2, "hidden_layer_nodes": 4},
        nuisance={"state": "disabled"},
    )

    model = DifferentiatingModel(
        context=context,
        detector_effect=_Detector(),
        is_numerator=True,
        name="disabled_nuisance",
        dtype=torch.float64,
    )

    assert isinstance(model.signal_region_shift_network, AdaptiveNeuralFunction)
    assert isinstance(model.nuisance_calculation, BlankNuisanceEstimator)


def test_canonical_binned_nuisance_uses_its_own_geometry():
    context = _context(
        f_options={"input_dimension": 2, "hidden_layer_nodes": 4},
        nuisance={
            "family": "bin_indicators",
            "options": {
                "minima": [0.0, -1.0],
                "maxima": [2.0, 1.0],
                "number_of_bins": [3, 4],
            },
        },
    )

    nuisance = build_nuisance_calculation(
        config=context.config,
        detector_effect=_Detector(),
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert isinstance(nuisance, ScalarBinnedNuisanceEstimator)
    assert isinstance(nuisance._bin_lookup, BinIndicatorFunction)
    assert nuisance._bin_lookup.geometry.number_of_bins == (3, 4)
    assert tuple(parameter.shape for parameter in nuisance._detector_deltas.values()) == (
        (3,),
        (4,),
    )

    model = DifferentiatingModel(
        context=context,
        detector_effect=_Detector(),
        is_numerator=True,
        name="different_role_families",
        dtype=torch.float64,
    )
    assert isinstance(model.signal_region_shift_network, AdaptiveNeuralFunction)
    assert isinstance(model.nuisance_calculation, ScalarBinnedNuisanceEstimator)


def test_unsupported_role_family_fails_at_adapter_boundary():
    context = _context(
        f_options={"input_dimension": 2, "hidden_layer_nodes": 4},
        nuisance={
            "family": "cubic_bspline",
            "options": {"knots": [0.0, 1.0]},
        },
    )

    with pytest.raises(ValueError, match="only adaptive_neural and bin_indicators"):
        build_nuisance_calculation(
            config=context.config,
            detector_effect=_Detector(),
            dtype=torch.float64,
            device=torch.device("cpu"),
        )


def test_backend_is_separate_from_role_family_support():
    resolved = resolve_dual_role_config(
        backend="nplm",
        f={
            "family": "adaptive_neural",
            "options": {"input_dimension": 2, "hidden_layer_nodes": 4},
        },
        nuisance={"state": "disabled"},
    )
    assert resolved.backend.value == "nplm"
    assert resolved.f.family is FunctionSpaceFamily.ADAPTIVE_NEURAL
    assert resolved.nuisance.state.value == "disabled"
