"""Output-map numerics, role configuration, and checkpoint compatibility."""

import json
from pathlib import Path

import pytest
import torch

from data_tools.data_utils import ShiftAndNormalizationFactor
from neural_networks.function_spaces import create_function_space
from neural_networks.likelihood_parameterization import (
    LIKELIHOOD_SHIFT_BOUND,
    smoothly_bounded_likelihood_shift,
)
from test.function_space_cases import NEURAL_DEPTH_CONFIGS, NEURAL_OUTPUT_CONFIGS
from train.checkpoint_metadata import (
    build_checkpoint_metadata,
    validate_checkpoint_metadata,
)
from train.neural_output_config import NeuralOutputConfig, NeuralOutputMapping
from train.train_config import TrainConfig


@pytest.mark.parametrize("mapping", list(NeuralOutputMapping))
def test_output_mapping_enum_normalizes_strings_and_preserves_serialized_names(mapping):
    config = NeuralOutputConfig(output_mapping=f" {mapping.value.upper()} ")
    assert config.output_mapping is mapping
    assert NeuralOutputConfig(output_mapping=mapping).output_mapping is mapping
    assert json.loads(json.dumps({"output_mapping": mapping})) == {
        "output_mapping": mapping.value
    }
    raw = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float64)
    torch.testing.assert_close(
        smoothly_bounded_likelihood_shift(raw, mapping),
        smoothly_bounded_likelihood_shift(raw, mapping.value),
    )


@pytest.mark.parametrize("mapping", ["algebraic", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_map_values_and_derivatives(mapping, dtype):
    raw = torch.tensor([-3.0, -0.2, 0.0, 0.2, 3.0], dtype=dtype, requires_grad=True)
    output = smoothly_bounded_likelihood_shift(raw, mapping)
    if mapping == "algebraic":
        expected = LIKELIHOOD_SHIFT_BOUND * raw / torch.sqrt(1 + raw.square())
        derivative = LIKELIHOOD_SHIFT_BOUND / (1 + raw.square()).pow(1.5)
    else:
        expected = LIKELIHOOD_SHIFT_BOUND * raw.tanh()
        derivative = LIKELIHOOD_SHIFT_BOUND / raw.cosh().square()
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    torch.testing.assert_close(raw.grad, derivative)
    assert torch.all(output.abs() < LIKELIHOOD_SHIFT_BOUND)
    torch.testing.assert_close(output, -output.flip(0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_algebraic_map_retains_gradient_at_observed_cluster_raw_outputs(dtype):
    raw = torch.tensor([-103.0, -60.0, 60.0, 103.0], dtype=dtype, requires_grad=True)
    output = smoothly_bounded_likelihood_shift(raw, "algebraic")
    output.sum().backward()
    assert torch.all(output.abs() < LIKELIHOOD_SHIFT_BOUND)
    assert torch.all(raw.grad > 0)
    assert torch.all(torch.tanh(raw).abs() == 1)


@pytest.mark.parametrize("mapping", ["algebraic", "tanh"])
def test_output_map_handles_extreme_finite_inputs_without_overflow(mapping):
    maximum = torch.finfo(torch.float64).max
    raw = torch.tensor([-maximum, 0.0, maximum], dtype=torch.float64)
    output = smoothly_bounded_likelihood_shift(raw, mapping)
    assert torch.isfinite(output).all()
    assert torch.all(output.abs() <= LIKELIHOOD_SHIFT_BOUND)
    assert torch.isfinite(torch.log1p(output)).all()
    assert torch.isfinite(torch.log1p(-output)).all()


@pytest.mark.parametrize(
    "function_execution_context, mappings",
    [
        (NEURAL_DEPTH_CONFIGS[0], ("algebraic", "algebraic")),
        *zip(
            NEURAL_OUTPUT_CONFIGS,
            [("algebraic", "algebraic"), ("tanh", "tanh"), ("algebraic", "tanh")],
        ),
    ],
    indirect=["function_execution_context"],
)
def test_file_backed_output_maps_are_independent_by_role(
    function_execution_context, mappings
):
    config = function_execution_context.config
    resolved = config.train__function_space_config
    for spec, mapping in zip((resolved.f, resolved.nuisance), mappings):
        space = create_function_space(
            spec,
            dtype=torch.float64,
            observable_names=config.detector__detect_observable_names,
        )
        assert space.output_config.output_mapping is NeuralOutputMapping(mapping)


@pytest.mark.parametrize("mapping", [None, True, 1, "sigmoid", "", [], {}])
@pytest.mark.parametrize("role", ["f", "nuisance"])
def test_invalid_output_map_is_rejected_for_either_role(mapping, role):
    specs = {
        "train__f": {"family": "adaptive_neural", "options": {"hidden_layer_nodes": 4}},
        "train__nuisance": {
            "family": "adaptive_neural",
            "options": {"hidden_layer_nodes": 2},
        },
    }
    specs[f"train__{role}"]["options"]["output_mapping"] = mapping
    with pytest.raises(ValueError, match="output_mapping"):
        TrainConfig(train__epochs=3, train__number_of_epochs_for_checkpoint=1, **specs)


def test_nplm_does_not_silently_ignore_lfvddp_output_mapping():
    with pytest.raises(ValueError, match="only supported by LFVDDP"):
        TrainConfig(
            train__epochs=3,
            train__number_of_epochs_for_checkpoint=1,
            train__backend="nplm",
            train__f={
                "family": "adaptive_neural",
                "options": {"hidden_layer_nodes": 4, "output_mapping": "tanh"},
            },
        )


@pytest.mark.parametrize(
    "function_execution_context", NEURAL_OUTPUT_CONFIGS[:2], indirect=True
)
def test_legacy_checkpoint_requires_explicit_tanh(function_execution_context):
    path = Path("test/configs/checkpoints/legacy_neural_widths.json")
    actual = json.loads(path.read_text())
    resolved = function_execution_context.config.train__function_space_config
    expected = build_checkpoint_metadata(
        model_name=actual["model_name"],
        is_numerator=actual["is_numerator"],
        resolved_config=resolved,
        normalization_factor=ShiftAndNormalizationFactor.from_mapping(
            actual["normalization_factor"]
        ),
    )
    kwargs = {
        "checkpoint_path": path,
        "model_name": actual["model_name"],
        "expected": expected,
        "actual": actual,
    }
    if resolved.f.options["output_mapping"] == "tanh":
        validate_checkpoint_metadata(**kwargs)
    else:
        with pytest.raises(RuntimeError, match="incompatible"):
            validate_checkpoint_metadata(**kwargs)


@pytest.mark.parametrize("role", ["f", "nuisance"])
@pytest.mark.parametrize(
    "function_execution_context", NEURAL_OUTPUT_CONFIGS[:2], indirect=True
)
def test_checkpoint_rejects_output_map_changes(function_execution_context, role):
    resolved = function_execution_context.config.train__function_space_config
    expected = build_checkpoint_metadata(
        model_name="mapping",
        is_numerator=True,
        resolved_config=resolved,
        normalization_factor=None,
    )
    actual = json.loads(json.dumps(expected))
    original = actual[role]["options"]["output_mapping"]
    actual[role]["options"]["output_mapping"] = (
        "tanh" if original == "algebraic" else "algebraic"
    )
    with pytest.raises(RuntimeError, match=f"incompatible.*{role}"):
        validate_checkpoint_metadata(
            checkpoint_path=Path("checkpoint.pt"),
            model_name="mapping",
            expected=expected,
            actual=actual,
        )
