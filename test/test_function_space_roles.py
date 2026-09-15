from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_tools.data_utils import DataSet
from neural_networks.differentiating_model import DifferentiatingModel
from neural_networks.function_spaces import (
    AdaptiveNeuralFunction,
    BinIndicatorFunction,
    CubicBSplineFunction,
    FixedSigmoidFunction,
    FUNCTION_SPACE_REGISTRY,
)
from neural_networks.nuisance_calculation import (
    BlankNuisanceEstimator,
    PerEventNuisanceEstimator,
    ScalarBinnedNuisanceEstimator,
    build_nuisance_calculation,
)
from train.function_space_config import (
    FunctionSpaceFamily,
    FunctionSpaceSpec,
    resolve_dual_role_config,
)
from train.train_config import TrainConfig


class _Detector:
    observable_names = ("x", "y")


def _context(*, f_options, nuisance, f_family="adaptive_neural"):
    config = TrainConfig(
        train__epochs=1,
        train__number_of_epochs_for_checkpoint=1,
        train__nn_inner_layer_nodes=2,
        train__nn_input_dimension=2,
        train__f={"family": f_family, "options": f_options},
        train__nuisance=nuisance,
    )
    return SimpleNamespace(config=config)


@pytest.mark.parametrize(
    "f",
    [
        {"family": "adaptive_neural", "options": {}},
        FunctionSpaceSpec(FunctionSpaceFamily.ADAPTIVE_NEURAL, {}),
    ],
    ids=["mapping", "typed-spec"],
)
def test_model_resolves_legacy_adaptive_f_defaults_in_canonical_config(f):
    config = TrainConfig(
        train__epochs=1,
        train__number_of_epochs_for_checkpoint=1,
        train__nn_inner_layer_nodes=3,
        train__nn_input_dimension=2,
        train__f=f,
        train__nuisance={
            "family": "adaptive_neural",
            "options": {"input_dimension": 2, "hidden_layer_nodes": 2},
        },
    )

    resolved = config.resolve_function_space_config()
    model = DifferentiatingModel(
        context=SimpleNamespace(config=config),
        detector_effect=_Detector(),
        is_numerator=True,
        name="legacy_adaptive_f",
        dtype=torch.float64,
    )

    assert dict(resolved.f.options) == {"input_dimension": 2, "hidden_size": 3}
    assert isinstance(model.signal_region_shift_network, AdaptiveNeuralFunction)
    assert model.signal_region_shift_network.input_dimension == 2
    assert model.signal_region_shift_network.hidden_size == 3


def test_adaptive_default_resolution_preserves_options_validation():
    with pytest.raises(ValueError, match="f.options must be a mapping"):
        TrainConfig(
            train__epochs=1,
            train__number_of_epochs_for_checkpoint=1,
            train__nn_inner_layer_nodes=3,
            train__f={"family": "adaptive_neural", "options": []},
            train__nuisance={
                "family": "adaptive_neural",
                "options": {"input_dimension": 1, "hidden_layer_nodes": 2},
            },
        )


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
    assert isinstance(model.nuisance_calculation, PerEventNuisanceEstimator)
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


def test_model_builds_fixed_family_for_both_roles():
    context = _context(
        f_options={"centers": [[0.0, 0.0], [1.0, 1.0]], "widths": [[0.5, 0.5], [0.5, 0.5]]},
        f_family="fixed_sigmoid",
        nuisance={
            "family": "fixed_sigmoid",
            "options": {
                "centers": [[-1.0, -1.0], [1.0, 1.0]],
                "widths": [[0.25, 0.25], [0.25, 0.25]],
            },
        },
    )

    model = DifferentiatingModel(
        context=context,
        detector_effect=_Detector(),
        is_numerator=True,
        name="fixed_roles",
        dtype=torch.float64,
    )

    assert isinstance(model.signal_region_shift_network, FixedSigmoidFunction)
    assert isinstance(model.nuisance_calculation, PerEventNuisanceEstimator)
    events = torch.tensor([[0.0, 0.5], [1.0, -0.5]], dtype=torch.float64)
    assert model.signal_region_shift_network(events).shape == (2, 1)
    assert model.nuisance_calculation.network(events).shape == (2, 1)


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
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert isinstance(nuisance, ScalarBinnedNuisanceEstimator)
    assert isinstance(nuisance._bin_lookup, BinIndicatorFunction)
    assert nuisance._bin_lookup.geometry.number_of_bins == (3, 4)
    assert tuple(parameter.shape for parameter in nuisance._bin_lookup._factor_deltas.values()) == (
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


def test_deterministic_family_uses_the_shared_nuisance_adapter():
    context = _context(
        f_options={"input_dimension": 2, "hidden_layer_nodes": 4},
        nuisance={
            "family": "cubic_bspline",
            "options": {"knots": [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]},
        },
    )

    nuisance = build_nuisance_calculation(
        config=context.config,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    assert isinstance(nuisance, PerEventNuisanceEstimator)
    assert isinstance(nuisance.network, CubicBSplineFunction)


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


def test_binned_nuisance_loads_shared_interface_checkpoint_parameter_names():
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
    model = DifferentiatingModel(
        context=context,
        detector_effect=_Detector(),
        is_numerator=True,
        name="shared_interface_binned_nuisance",
        dtype=torch.float64,
    )
    shared_interface_state = {
        key.replace(
            "nuisance_calculation._nuisance_deltas.",
            "nuisance_calculation._bin_lookup._factor_deltas.",
        ): value.clone()
        for key, value in model.state_dict().items()
    }

    restored = DifferentiatingModel(
        context=context,
        detector_effect=_Detector(),
        is_numerator=True,
        name="restored_shared_interface_binned_nuisance",
        dtype=torch.float64,
    )
    restored.load_state_dict(shared_interface_state, strict=True)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


_ONE_DIMENSIONAL_FAMILY_OPTIONS = {
    FunctionSpaceFamily.ADAPTIVE_NEURAL: {
        "input_dimension": 1,
        "hidden_layer_nodes": 3,
    },
    FunctionSpaceFamily.BIN_INDICATORS: {
        "minima": [-2.0],
        "maxima": [2.0],
        "number_of_bins": [3],
    },
    FunctionSpaceFamily.CUBIC_BSPLINE: {
        "knots": [-2.0, -1.0, 0.0, 1.0, 2.0],
    },
    FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL: {
        "basis": "legendre",
        "maximum_degree": 2,
        "domain": [-2.0, 2.0],
    },
    FunctionSpaceFamily.FIXED_SIGMOID: {
        "centers": [-1.0, 1.0],
        "widths": [0.5, 0.5],
    },
    FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS: {
        "centers": [-1.0, 1.0],
        "widths": [0.5, 0.5],
    },
}


@pytest.mark.parametrize(
    ("family", "options"),
    tuple(_ONE_DIMENSIONAL_FAMILY_OPTIONS.items()),
    ids=lambda case: case.value if isinstance(case, FunctionSpaceFamily) else None,
)
def test_every_registered_family_supports_independent_roles_nuisance_training_and_prediction(
    family,
    options,
):
    context = _context(
        f_family=family,
        f_options=options,
        nuisance={"family": family, "options": options},
    )
    model = DifferentiatingModel(
        context=context,
        detector_effect=_Detector(),
        is_numerator=True,
        name=f"{family.value}_roles",
        dtype=torch.float64,
    )
    nuisance = model.nuisance_calculation

    assert set(_ONE_DIMENSIONAL_FAMILY_OPTIONS) == set(FUNCTION_SPACE_REGISTRY)
    if family is FunctionSpaceFamily.BIN_INDICATORS:
        assert isinstance(nuisance, ScalarBinnedNuisanceEstimator)
        assert nuisance._bin_lookup is not model.signal_region_shift_network
    else:
        assert isinstance(nuisance, PerEventNuisanceEstimator)
        assert nuisance.network is not model.signal_region_shift_network
    assert {
        id(parameter) for parameter in model.signal_region_shift_network.parameters()
    }.isdisjoint({id(parameter) for parameter in nuisance.parameters()})

    signal_region = DataSet(
        np.array([[-1.0], [0.0], [1.0]]), observable_names=["x"]
    )
    a_control_region = DataSet(
        np.array([[-0.75], [0.25]]), observable_names=["x"]
    )
    b_control_region = DataSet(
        np.array([[-0.25], [0.75]]), observable_names=["x"]
    )
    normalized_signal_region, normalization_factor = signal_region.get_normalized()
    prepared = nuisance.prepare(
        signal_region,
        a_control_region,
        b_control_region,
        normalized_signal_region,
        a_control_region / normalization_factor,
        b_control_region / normalization_factor,
    )
    evaluation = nuisance.evaluate(prepared)
    nuisance_loss = (
        evaluation.nuisance_sr_values.sum()
        + evaluation.nuisance_cr_a.values.sum()
        + evaluation.nuisance_cr_b.values.sum()
    )
    assert torch.isfinite(nuisance_loss)

    nuisance_parameters = tuple(nuisance.parameters())
    before_step = tuple(parameter.detach().clone() for parameter in nuisance_parameters)
    optimizer = torch.optim.SGD(nuisance_parameters, lr=0.1)
    optimizer.zero_grad(set_to_none=True)
    nuisance_loss.backward()
    assert any(parameter.grad is not None for parameter in nuisance_parameters)
    optimizer.step()
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_step, nuisance_parameters)
    )

    model._norm_factor = normalization_factor
    primary = model.predict(signal_region)
    secondary = model.predict_secondary(signal_region)
    nuisance_prediction = model.predict_theta(signal_region)
    assert primary.shape == secondary.shape == nuisance_prediction.shape == (3, 1)
    assert np.isfinite(primary).all()
    assert np.isfinite(secondary).all()
    assert np.isfinite(nuisance_prediction).all()
