"""File-backed coverage for configurable neural depth and diagnostic counts."""

from math import sqrt

import numpy as np
import pytest
import torch
from torch import nn

from data_tools.data_utils import DataSet
from frame.aggregate import ResultAggregator
from frame.command_line.handle_args import create_config_from_paths
from neural_networks.function_spaces import (
    analytic_degrees_of_freedom,
    create_function_space,
)
from neural_networks.likelihood_parameterization import (
    smoothly_bounded_likelihood_shift,
)
from test.environment import DEFAULT_CONFIG_PATHS, ConfigType
from test.function_space_cases import NEURAL_DEPTH_CASES, NEURAL_DEPTH_CONFIGS
from train.function_space_config import FunctionSpaceSpec
from train.train_config import TrainConfig


@pytest.mark.parametrize(
    "function_execution_context, architecture, parameter_count",
    [
        pytest.param(config, case[2], case[3], id=case[0])
        for config, case in zip(NEURAL_DEPTH_CONFIGS, NEURAL_DEPTH_CASES)
    ],
    indirect=["function_execution_context"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_neural_topology_initialization_gradients_and_count(
    function_execution_context,
    architecture,
    parameter_count,
    dtype,
):
    config = function_execution_context.config
    spec = config.train__function_space_config.f
    space = create_function_space(
        spec,
        dtype=dtype,
        device=torch.device("cpu"),
        observable_names=config.detector__detect_observable_names,
    )
    assert space.architecture == architecture
    assert config.train__adaptive_architecture == list(architecture)
    assert isinstance(space.activation, nn.Sigmoid)
    assert len(space.hidden_layers) == len(architecture) - 2
    assert space.statistical_degrees_of_freedom() == parameter_count
    assert (
        space.analytic_degrees_of_freedom(
            spec.options, observable_count=config.detector__number_of_dimensions
        )
        == parameter_count
    )
    assert (
        "input_dimension" not in spec.options and "output_dimension" not in spec.options
    )
    assert sum(parameter.numel() for parameter in space.parameters()) == parameter_count

    gain = 0.7
    space.initialize_parameters(gain)
    layers = (*space.hidden_layers, space.output)
    for layer, source, destination in zip(layers, architecture, architecture[1:]):
        assert (layer.in_features, layer.out_features) == (source, destination)
        assert layer.weight.dtype == layer.bias.dtype == dtype
        assert layer.weight.device.type == "cpu"
        assert torch.all(layer.weight.abs() <= gain * sqrt(6 / (source + destination)))
        assert torch.all(layer.bias.abs() <= 0.3)

    events = torch.linspace(-0.5, 0.7, steps=3 * architecture[0], dtype=dtype).reshape(
        3, -1
    )
    expected = events
    for layer in space.hidden_layers:
        expected = torch.sigmoid(
            torch.nn.functional.linear(expected, layer.weight, layer.bias)
        )
    expected = smoothly_bounded_likelihood_shift(
        torch.nn.functional.linear(expected, space.output.weight, space.output.bias)
    )
    actual = space(events)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad)
        for parameter in space.parameters()
    )
    if not space.hidden_layers:
        assert tuple(space.state_dict()) == ("output.weight", "output.bias")


@pytest.mark.parametrize(
    "function_execution_context", NEURAL_DEPTH_CONFIGS, indirect=True
)
def test_neural_depth_training_all_predictions_and_denominator(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    data = batch.datasets[DataSet.DataSetCategory.A_SR]
    for is_numerator in (True, False):
        model = differentiating_model_factory(
            function_execution_context,
            detector_effect,
            is_numerator=is_numerator,
            name=f"depth_{is_numerator}",
        )
        history = model.fit(batch)
        assert history["loss"] and np.isfinite(history["loss"]).all()
        assert model.nuisance_function_space is not None
        predictions = [
            model.predict_theta(data),
            model.predict(data),
            model.predict_secondary(data),
        ]
        for prediction in predictions:
            assert prediction.shape == (data.n_samples, 1)
            assert np.isfinite(prediction).all()


def test_integer_and_single_list_preserve_legacy_state_and_random_order():
    from frame.file_system.textual_data import load_config_params_from_paths
    from train.function_space_config import FunctionSpaceSpec

    spaces = []
    for config in NEURAL_DEPTH_CONFIGS[:2]:
        parameters = load_config_params_from_paths([config[ConfigType.TRAIN]])
        spec = FunctionSpaceSpec(**parameters["train__f"])
        torch.manual_seed(812)
        space = create_function_space(
            spec, dtype=torch.float64, observable_names=("param_0",)
        )
        space.initialize_parameters(gain=0.7)
        spaces.append(space)
    integer_space, list_space = spaces
    expected_keys = ("hidden.weight", "hidden.bias", "output.weight", "output.bias")
    assert (
        tuple(integer_space.state_dict())
        == tuple(list_space.state_dict())
        == expected_keys
    )
    for key, parameter in integer_space.state_dict().items():
        torch.testing.assert_close(
            list_space.state_dict()[key], parameter, rtol=0, atol=0
        )

    # Reproduce the historical two-layer construction and initialization order.
    torch.manual_seed(812)
    hidden = nn.Linear(1, 4, dtype=torch.float64)
    output = nn.Linear(4, 1, dtype=torch.float64)
    for layer in (hidden, output):
        nn.init.xavier_uniform_(layer.weight, gain=0.7)
        nn.init.uniform_(layer.bias, a=-0.3, b=0.3)
    for name, layer in (("hidden", hidden), ("output", output)):
        torch.testing.assert_close(
            integer_space.state_dict()[f"{name}.weight"], layer.weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            integer_space.state_dict()[f"{name}.bias"], layer.bias, rtol=0, atol=0
        )
    list_space.load_state_dict(integer_space.state_dict(), strict=True)
    events = torch.tensor([[-1.0], [0.1], [1.0]], dtype=torch.float64)
    expected = smoothly_bounded_likelihood_shift(output(torch.sigmoid(hidden(events))))
    torch.testing.assert_close(list_space(events), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "widths",
    [True, False, 0, -1, 1.5, "4", None, [0], [-1], [True], [False], [2.5], [4, []]],
)
def test_invalid_hidden_widths_fail_configuration_validation(widths):
    with pytest.raises(ValueError, match="hidden_layer_nodes"):
        TrainConfig(
            train__epochs=3,
            train__number_of_epochs_for_checkpoint=1,
            train__f={
                "family": "adaptive_neural",
                "options": {"hidden_layer_nodes": widths},
            },
        )


@pytest.mark.parametrize(
    "field, value",
    [
        ("input_dimension", True),
        ("input_dimension", 0),
        ("input_dimension", 1.5),
        ("output_dimension", 2),
        ("output_dimension", True),
        ("output_dimension", 1.0),
    ],
)
def test_obsolete_dimensions_cannot_override_inferred_widths(field, value):
    config = TrainConfig(
        train__epochs=3,
        train__number_of_epochs_for_checkpoint=1,
        train__f={
            "family": "adaptive_neural",
            "options": {
                "hidden_layer_nodes": 4,
                field: value,
            },
        },
    )
    spec = config.train__function_space_config.f
    assert field not in spec.options
    space = create_function_space(
        spec, dtype=torch.float64, observable_names=("x", "y")
    )
    assert space.architecture == (2, 4, 1)


@pytest.mark.parametrize("widths", [[4], [4, 3], []])
def test_nplm_rejects_list_architectures(widths):
    with pytest.raises(ValueError, match="supported only by LFVDDP"):
        TrainConfig(
            train__epochs=3,
            train__number_of_epochs_for_checkpoint=1,
            train__backend="nplm",
            train__f={
                "family": "adaptive_neural",
                "options": {"hidden_layer_nodes": widths},
            },
        )


def test_network_input_must_match_detector_observables():
    paths = {
        **DEFAULT_CONFIG_PATHS,
        **NEURAL_DEPTH_CONFIGS[0],
        ConfigType.TRAIN: NEURAL_DEPTH_CONFIGS[4][ConfigType.TRAIN],
    }
    config = create_config_from_paths(list(paths.values()))
    assert config.train__adaptive_architecture == [1, 4, 3, 2, 1]
    for spec in (
        config.train__function_space_config.f,
        config.train__function_space_config.nuisance,
    ):
        space = create_function_space(
            spec,
            dtype=torch.float64,
            observable_names=config.detector__detect_observable_names,
        )
        assert space.input_dimension == 1 and space.output_dimension == 1


def test_neural_construction_and_count_require_observables():
    spec = FunctionSpaceSpec("adaptive_neural", {"hidden_layer_nodes": [4, 3]})
    with pytest.raises(ValueError, match="observable_names"):
        create_function_space(spec, dtype=torch.float64)
    with pytest.raises(ValueError, match="observable count"):
        analytic_degrees_of_freedom(spec)
    with pytest.raises(ValueError, match="positive input_dimension"):
        create_function_space(spec, dtype=torch.float64, observable_names=())


def test_neural_factory_consumes_observables_once_and_disallows_width_override():
    spec = FunctionSpaceSpec("adaptive_neural", {"hidden_layer_nodes": []})
    space = create_function_space(
        spec, dtype=torch.float64, observable_names=iter(("x", "y"))
    )
    assert space.architecture == (2, 1)
    with pytest.raises(TypeError, match="input_dimension"):
        create_function_space(
            spec, dtype=torch.float64, observable_names=("x", "y"), input_dimension=99
        )


@pytest.mark.parametrize("field", ["input_dimension", "output_dimension"])
def test_canonical_neural_factory_has_no_configurable_endpoint_width(field):
    spec = FunctionSpaceSpec("adaptive_neural", {"hidden_layer_nodes": 4, field: 1})
    with pytest.raises(ValueError, match="not configurable"):
        create_function_space(spec, dtype=torch.float64, observable_names=("x", "y"))


@pytest.mark.parametrize(
    "function_execution_context, expected_count",
    [
        pytest.param(config, case[3], id=case[0])
        for config, case in zip(NEURAL_DEPTH_CONFIGS, NEURAL_DEPTH_CASES)
    ],
    indirect=["function_execution_context"],
)
def test_aggregate_uses_signal_parameter_count_only(
    function_execution_context, expected_count, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "frame.aggregate.ExecutionContext.discover_run_contexts",
        lambda _: [(function_execution_context, tmp_path)],
    )
    assert ResultAggregator(tmp_path).chi_square_degrees_of_freedom == expected_count


@pytest.mark.parametrize(
    "function_execution_context", [NEURAL_DEPTH_CONFIGS[0]], indirect=True
)
def test_aggregate_rejects_mixed_neural_counts(
    function_execution_context, monkeypatch, tmp_path
):
    from types import SimpleNamespace

    paths = {**DEFAULT_CONFIG_PATHS, **NEURAL_DEPTH_CONFIGS[2]}
    other = SimpleNamespace(config=create_config_from_paths(list(paths.values())))
    monkeypatch.setattr(
        "frame.aggregate.ExecutionContext.discover_run_contexts",
        lambda _: [(function_execution_context, tmp_path), (other, tmp_path)],
    )
    with pytest.raises(ValueError, match="different effective test-statistic degrees"):
        _ = ResultAggregator(tmp_path).chi_square_degrees_of_freedom
