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
