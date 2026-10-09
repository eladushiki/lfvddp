"""Construction and configuration coverage shared by every function-space family."""

import pytest
import torch

from neural_networks.function_spaces import create_function_space
from neural_networks.function_spaces.registry import FUNCTION_SPACE_REGISTRY
from test.function_space_cases import FUNCTION_SPACE_OPTIONS
from train.function_space_config import FunctionSpaceSpec
from train.train_config import TrainConfig


def test_every_catalog_family_constructs_one_normalized_event_shift():
    assert set(FUNCTION_SPACE_OPTIONS) == set(FUNCTION_SPACE_REGISTRY)
    events = torch.tensor([[0.5], [1.5]], dtype=torch.float64)
    for family, options in FUNCTION_SPACE_OPTIONS.items():
        space = create_function_space(
            FunctionSpaceSpec(family, options), dtype=torch.float64
        )
        assert space(events).shape == (2, 1)


@pytest.mark.parametrize(
    "spec, message",
    [
        (
            {"family": "bin_indicators", "options": {"minima": [0], "maxima": [1]}},
            "Missing bin geometry option",
        ),
        (
            {"family": "fixed_sigmoid", "options": {"centers": [0], "widths": [0]}},
            "positive",
        ),
        (
            {
                "family": "orthogonal_polynomial",
                "options": {"basis": "fourier", "maximum_degree": 2, "domain": [0, 1]},
            },
            "PolynomialBasis",
        ),
        ({"family": "cubic_bspline", "options": {"knots": [0, 1, 1]}}, "knots"),
        ({"family": "gaussian_radial_basis", "options": {"centers": [0]}}, "requires"),
        (
            {
                "family": "bin_indicators",
                "options": {"minima": [0], "maxima": [1], "number_of_bins": [1.5]},
            },
            "integers",
        ),
        (
            {
                "family": "bin_indicators",
                "options": {
                    "minima": [0],
                    "maxima": [1],
                    "number_of_bins": [1],
                    "output_dimension": 2,
                },
            },
            "output_dimension",
        ),
        (
            {
                "family": "orthogonal_polynomial",
                "options": {
                    "basis": "legendre",
                    "maximum_degree": 2,
                    "domain": [0, 1],
                    "tensor_product_basis": "yes",
                },
            },
            "tensor_product_basis",
        ),
    ],
)
def test_family_options_fail_through_the_normal_configuration_path(spec, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(
            train__epochs=100,
            train__number_of_epochs_for_checkpoint=10,
            train__f=spec,
            train__nuisance=None,
        )


def test_adaptive_neural_requires_explicit_canonical_dimensions():
    with pytest.raises(ValueError, match="hidden_layer_nodes"):
        create_function_space(
            FunctionSpaceSpec("adaptive_neural", {"input_dimension": 1}),
            dtype=torch.float64,
        )
    with pytest.raises(ValueError, match="hidden_layer_nodes"):
        TrainConfig(
            train__epochs=100,
            train__number_of_epochs_for_checkpoint=10,
            train__f={
                "family": "adaptive_neural",
                "options": {"input_dimension": 1, "hidden_size": 2},
            },
            train__nuisance=None,
        )


@pytest.mark.parametrize(
    "widths", [None, [], 0, -1, True, [4, 0], [4, -2], [4, 2.5], [False], "4"]
)
def test_adaptive_hidden_widths_reject_invalid_values(widths):
    with pytest.raises(ValueError, match="hidden_layer_nodes"):
        create_function_space(
            FunctionSpaceSpec(
                "adaptive_neural",
                {
                    "input_dimension": 2,
                    "hidden_layer_nodes": widths,
                },
            ),
            dtype=torch.float64,
        )


@pytest.mark.parametrize("widths", [4, [4], [4, 2], (8, 2)])
def test_adaptive_hidden_widths_build_all_layers_and_nplm_architecture(widths):
    config = TrainConfig(
        train__epochs=100000,
        train__number_of_epochs_for_checkpoint=10000,
        train__backend="nplm",
        train__f={
            "family": "adaptive_neural",
            "options": {
                "input_dimension": 2,
                "hidden_layer_nodes": widths,
            },
        },
    )
    space = create_function_space(
        config.train__function_space_config.f, dtype=torch.float64
    )
    architecture = config.train__adaptive_architecture
    layers = [space.hidden, *space.additional_hidden, space.output]
    assert [(layer.in_features, layer.out_features) for layer in layers] == list(
        zip(architecture, architecture[1:])
    )
    space.initialize_parameters(1.0)
    assert all(torch.all(layer.bias.abs() <= 0.3) for layer in layers)
    assert space(torch.zeros((3, 2), dtype=torch.float64)).shape == (3, 1)
    if architecture == [2, 4, 1]:
        assert set(space.state_dict()) == {
            "hidden.weight",
            "hidden.bias",
            "output.weight",
            "output.bias",
        }
