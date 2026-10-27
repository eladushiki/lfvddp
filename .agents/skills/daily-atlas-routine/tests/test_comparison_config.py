"""Comparison guards: no PBS, model training or scientific plot generation."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from atlas_routine.backend import Backend

from train.comparison_config import (
    ComparisonConfigError,
    saved_comparison_configs,
    validate_comparison_configs,
)

CASES = json.loads((Path(__file__).parent / "configs/comparison.json").read_text())


@pytest.mark.parametrize("change", CASES["mismatches"])
def test_every_fitting_difference_blocks_even_a_late_signal_point(change):
    configs = [("background", CASES["base"]), ("signal01", CASES["base"])]
    configs.append(("signal05", CASES["base"] | change))
    with pytest.raises(ComparisonConfigError, match=next(iter(change))):
        validate_comparison_configs(configs)


def test_injection_seed_array_size_and_display_may_differ():
    validate_comparison_configs(
        [
            ("background", CASES["base"]),
            ("signal", CASES["base"] | CASES["allowed"]),
        ]
    )


@pytest.mark.parametrize("configs", [[], [("incomplete", {})]])
def test_missing_evidence_fails_closed(configs):
    with pytest.raises(ComparisonConfigError):
        validate_comparison_configs(configs)


def test_available_worker_override_is_checked(tmp_path):
    (tmp_path / "context.json").write_text(json.dumps({"config": CASES["base"]}))
    worker = tmp_path / "stamp_run_of_single_train.py_pid_1"
    worker.mkdir()
    (worker / "context.json").write_text(
        json.dumps({"config": CASES["base"] | {"train__epochs": 10000}})
    )
    with pytest.raises(ComparisonConfigError, match="train__epochs"):
        validate_comparison_configs(saved_comparison_configs(tmp_path))


def test_planned_packs_are_compared_with_saved_background(tmp_path):
    from frame.context.execution_context import ExecutionContext

    (tmp_path / "context.json").write_text(json.dumps({"config": CASES["base"]}))
    backend = Backend(tmp_path, tmp_path / "logs", {})
    background = {
        "id": "bkg",
        "status": "analyzed",
        "remote_submission_directory": str(tmp_path),
    }
    signal = {
        "id": "signal",
        "status": "requested",
        "remote_submission_directory": "superseded",
    }
    backend.configs = Mock(return_value=(Mock(), []))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            ExecutionContext,
            "serialize",
            lambda config: CASES["base"] | {"train__learning_rate": 0.1},
        )
        with pytest.raises(ComparisonConfigError, match="train__learning_rate"):
            backend.validate_comparison([background, signal])
    backend.configs.assert_called_once_with(signal)
    with pytest.raises(ValueError, match="Missing saved"):
        backend.validate_comparison([{"id": "missing"}], saved_only=True)


def test_real_enabled_nuisance_config_uses_the_same_resolved_signature(tmp_path):
    from frame.context.execution_context import ExecutionContext

    root = Path(__file__).resolve().parents[4]
    backend = Backend(root, tmp_path, {})
    entry = {
        "id": "planned",
        "status": "requested",
        "output_root": str(tmp_path),
        "config_packs": [
            "configs/basic-generated",
            "test/configs/train/adaptive_neural_nuisance.json",
        ],
    }
    config, _ = backend.configs(entry)
    assert config.train__function_space_config.nuisance.family == "adaptive_neural"
    # Use the same persisted JSON boundary as real submission/worker contexts.
    (tmp_path / "context.json").write_text(
        json.dumps({"config": ExecutionContext.serialize(config)})
    )
    saved = {
        "id": "saved",
        "status": "analyzed",
        "remote_submission_directory": str(tmp_path),
    }
    backend.validate_comparison([saved, entry])
