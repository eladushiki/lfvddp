"""Dataset comparison and signal-curve identity without data loading or sampling."""

import json
from pathlib import Path

import pytest

from train.comparison_config import ComparisonConfigError, validate_comparison_configs
from train.comparison_datasets import (
    SIGNAL_SHAPE_FIELDS,
    SIGNAL_STRENGTH_FIELDS,
    dataset_comparison_values,
    signal_curve_key,
)

FIXTURES = Path(__file__).parent / "configs"
ROOT = Path(__file__).resolve().parents[4]
BASE = json.loads((FIXTURES / "comparison.json").read_text())["base"]
CASES = json.loads((FIXTURES / "comparison_datasets.json").read_text())


@pytest.mark.parametrize(
    "source,change",
    [(case["source"], change) for case in CASES.values() for change in case["changes"]],
)
def test_non_signal_dataset_difference_blocks(source, change):
    config = BASE | json.loads((ROOT / source).read_text())
    changed = config | {
        "dataset__definitions": [
            config["dataset__definitions"][0] | change,
            *config["dataset__definitions"][1:],
        ]
    }
    with pytest.raises(ComparisonConfigError, match=next(iter(change))):
        validate_comparison_configs([("background", config), ("signal", changed)])


@pytest.mark.parametrize("case", CASES.values())
def test_signal_free_reference_and_reordered_regions_are_compatible(case, monkeypatch):
    config = BASE | json.loads((ROOT / case["source"]).read_text())
    reference = config | {
        "dataset__definitions": [
            {
                k: v
                for k, v in d.items()
                if k not in SIGNAL_SHAPE_FIELDS | SIGNAL_STRENGTH_FIELDS
            }
            for d in reversed(config["dataset__definitions"])
        ]
    }
    # Comparability must not realize event counts or touch the ROOT file.
    monkeypatch.setattr(
        "numpy.random.poisson",
        lambda *a, **kw: pytest.fail("sampled during validation"),
    )
    validate_comparison_configs([("background", reference), ("signal", config)])


def test_strength_changes_stay_in_curve_but_shape_changes_split_curves():
    definitions = BASE["dataset__definitions"]
    strength = BASE | {
        "dataset__definitions": [
            definitions[0] | {"dataset__mean_number_of_signal_events": 200},
            *definitions[1:],
        ]
    }
    shape = BASE | {
        "dataset__definitions": [
            definitions[0] | {"dataset__signal_positive_only": False},
            *definitions[1:],
        ]
    }
    generator = BASE | {
        "dataset__definitions": [
            definitions[0]
            | {
                "dataset__signal_generator": {
                    "function": "gaussian_signal",
                    "arguments": {"location": 3, "gaussian_signal_sigma": 0.8},
                }
            },
            *definitions[1:],
        ]
    }
    assert signal_curve_key(BASE) == signal_curve_key(strength)
    assert signal_curve_key(BASE) != signal_curve_key(shape)
    assert signal_curve_key(BASE) != signal_curve_key(generator)
    validate_comparison_configs(
        [
            ("background", BASE),
            ("strength", strength),
            ("shape", shape),
            ("generator", generator),
        ]
    )


@pytest.mark.parametrize(
    "definitions",
    [
        [],
        [{"category": "a_sr", "type": "generated"}],
        [BASE["dataset__definitions"][0]] * 2,
    ],
)
def test_missing_or_duplicate_dataset_evidence_is_rejected(definitions):
    with pytest.raises((ValueError, KeyError)):
        dataset_comparison_values({"dataset__definitions": definitions})


def test_omitted_dataset_defaults_equal_explicit_defaults():
    definitions = BASE["dataset__definitions"]
    explicit = BASE | {
        "dataset__definitions": [
            definitions[0] | {"dataset__induced_shape_nuisance_value": 0.0},
            *definitions[1:],
        ]
    }
    validate_comparison_configs([("omitted", BASE), ("explicit", explicit)])


def test_loaded_signal_observable_mapping_is_shape_not_background_configuration():
    config = BASE | json.loads((ROOT / CASES["loaded"]["source"]).read_text())
    definitions = config["dataset__definitions"]
    mapped = config | {
        "dataset__definitions": [
            definitions[0]
            | {"dataset_loaded__signal_observable_names": ["Lepton_mass"]},
            *definitions[1:],
        ]
    }
    validate_comparison_configs([("reference", config), ("mapped signal", mapped)])
    assert signal_curve_key(config) != signal_curve_key(mapped)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("saved_only", [False, True])
def test_resampling_on_off_warns_for_planned_and_saved_runs(
    tmp_path, monkeypatch, reverse, saved_only
):
    from unittest.mock import Mock

    from atlas_routine.backend import Backend

    from frame.context.execution_context import ExecutionContext

    config = BASE | json.loads((ROOT / CASES["loaded"]["source"]).read_text())
    unresampled = config | {
        "dataset__definitions": [
            definition | {"dataset_loaded__resample_is_resample": False}
            for definition in config["dataset__definitions"]
        ]
    }
    reference, candidate = (unresampled, config) if reverse else (config, unresampled)
    assert signal_curve_key(reference) != signal_curve_key(candidate)
    background_dir = tmp_path / "background"
    signal_dir = tmp_path / "signal"
    background_dir.mkdir()
    signal_dir.mkdir()
    for directory, values in [(background_dir, reference), (signal_dir, candidate)]:
        (directory / "context.json").write_text(json.dumps({"config": values}))
    worker = signal_dir / "stamp_run_of_single_train.py_pid_1"
    worker.mkdir()
    (worker / "context.json").write_text(json.dumps({"config": candidate}))
    backend = Backend(tmp_path, tmp_path / "logs", {})
    backend.configs = Mock(return_value=(Mock(), []))
    monkeypatch.setattr(ExecutionContext, "serialize", lambda config: candidate)
    members = [
        {
            "id": "reference",
            "status": "analyzed",
            "remote_submission_directory": str(background_dir),
        },
        {
            "id": "candidate",
            "status": "analyzed" if saved_only else "requested",
            "remote_submission_directory": str(signal_dir),
        },
    ]
    with pytest.warns(UserWarning, match="resampling on/off") as caught:
        backend.validate_comparison(members, saved_only=saved_only)
    assert len(caught) == 1
    assert "dataset_loaded__resample_is_resample" in str(caught[0].message)


@pytest.mark.parametrize("change", CASES["loaded"]["changes"])
def test_resampling_exception_does_not_hide_other_dataset_mismatches(change):
    config = BASE | json.loads((ROOT / CASES["loaded"]["source"]).read_text())
    candidate = config | {
        "dataset__definitions": [
            definition | {"dataset_loaded__resample_is_resample": False} | change
            for definition in config["dataset__definitions"]
        ]
    }
    with pytest.raises(ComparisonConfigError, match=next(iter(change))):
        validate_comparison_configs([("reference", config), ("candidate", candidate)])


@pytest.mark.parametrize(
    "change",
    [
        {"train__epochs": 1},
        {"detector__effect_b_error": 0.2},
    ],
)
def test_resampling_exception_does_not_hide_fitting_mismatches(change):
    config = BASE | json.loads((ROOT / CASES["loaded"]["source"]).read_text())
    candidate = (
        config
        | change
        | {
            "dataset__definitions": [
                definition | {"dataset_loaded__resample_is_resample": False}
                for definition in config["dataset__definitions"]
            ]
        }
    )
    with pytest.raises(ComparisonConfigError, match=next(iter(change))):
        validate_comparison_configs([("reference", config), ("candidate", candidate)])


@pytest.mark.parametrize("value", [None, "false", 0])
def test_non_boolean_resampling_difference_is_not_exempted(value):
    config = BASE | json.loads((ROOT / CASES["loaded"]["source"]).read_text())
    candidate = config | {
        "dataset__definitions": [
            definition | {"dataset_loaded__resample_is_resample": value}
            for definition in config["dataset__definitions"]
        ]
    }
    with pytest.raises(
        ComparisonConfigError, match="dataset_loaded__resample_is_resample"
    ):
        validate_comparison_configs([("reference", config), ("candidate", candidate)])
