import pickle
from pathlib import Path

import numpy as np
import pytest

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet, ShiftAndNormalizationFactor
from data_tools.dataset_config import (
    DatasetConfig,
    GeneratedDatasetParameters,
)
from test.environment import ConfigType


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/adaptive_neural_nuisance.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
        }
    ],
    indirect=True,
)
def test_dataset_normalization(
    function_execution_context,
    data_generation,
):
    A, _ = data_generation[DataSet.DataSetCategory.A_SR]
    B, _ = data_generation[DataSet.DataSetCategory.B_SR]

    normalized_A, norm_factor_A = A.get_normalized()
    normalized_B, norm_factor_B = B.get_normalized()

    for raw, normalized in ((A, normalized_A), (B, normalized_B)):
        if np.ptp(raw.events) == 0:
            np.testing.assert_allclose(normalized.events, 0.0)
        else:
            np.testing.assert_allclose(
                np.mean(normalized.events, axis=0), 0, rtol=1e-9, atol=1e-9
            )
            np.testing.assert_allclose(
                np.std(normalized.events, axis=0), 1, rtol=1e-9, atol=1e-9
            )

    np.testing.assert_array_almost_equal(
        (normalized_A * norm_factor_A).events, A.events
    )
    np.testing.assert_array_almost_equal(
        (normalized_B * norm_factor_B).events, B.events
    )


def test_dataset_normalization_maps_constant_observables_to_zero():
    dataset = DataSet(
        np.array(
            [
                [2.0, 1.0],
                [4.0, 1.0],
                [6.0, 1.0],
            ]
        ),
        observable_names=["varying", "constant"],
    )

    normalized, normalization_factor = dataset.get_normalized()

    assert np.isfinite(normalized.events).all()
    np.testing.assert_allclose(
        normalized.events[:, 0], np.array([-1.0, 0.0, 1.0]) * np.sqrt(1.5)
    )
    np.testing.assert_allclose(normalized.events[:, 1], 0.0)
    np.testing.assert_allclose(
        (normalized * normalization_factor).events,
        dataset.events,
    )


def test_batch_normalization_uses_one_pooled_affine_map_for_every_category():
    categories = DataSet.DataSetCategory
    raw_datasets = {
        categories.A_SR: DataSet(np.array([[0.0], [2.0]]), ["x"], categories.A_SR),
        categories.B_SR: DataSet(np.array([[8.0], [10.0]]), ["x"], categories.B_SR),
        categories.A_CR: DataSet(np.array([[1.0], [3.0]]), ["x"], categories.A_CR),
        categories.B_CR: DataSet(np.array([[7.0], [9.0]]), ["x"], categories.B_CR),
    }
    batch = DataBatch((dataset, None) for dataset in raw_datasets.values())

    normalized, normalization_factor = batch.get_normalized()

    assert normalization_factor.get_offset("x") == 5.0
    assert normalization_factor.get_factor("x") == np.sqrt(13.5)
    expected = normalization_factor.normalize_values(
        np.concatenate(
            [
                raw_datasets[category].events
                for category in DataBatch.REQUIRED_DATASET_CATEGORIES
            ]
        ),
        ("x",),
    )
    actual = np.concatenate([dataset.events for dataset, _ in normalized])
    np.testing.assert_allclose(actual, expected)
    pooled, _ = batch.unified_data.get_normalized()
    np.testing.assert_allclose(actual, pooled.events)
    np.testing.assert_allclose(actual.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(actual.std(axis=0), 1.0)
    for category, raw_dataset in raw_datasets.items():
        np.testing.assert_allclose(
            (normalized.datasets[category] * normalization_factor).events,
            raw_dataset.events,
        )


def test_dataset_config_owns_signal_configuration_fields():
    assert DatasetConfig.SIGNAL_EVENT_CONFIGURATION_FIELDS == {
        "dataset__mean_number_of_signal_events",
    }
    assert DatasetConfig.SIGNAL_CONFIGURATION_FIELDS == {
        "dataset__mean_number_of_signal_events",
        "dataset__signal_generator",
        "dataset__signal_positive_only",
    }


@pytest.mark.parametrize("constant", [0.1, 1.0, 1e20])
def test_standardization_is_per_observable_and_keeps_constant_inputs_finite(constant):
    dataset = DataSet(
        np.array(
            [[1.0, 100.0, constant], [3.0, 400.0, constant], [8.0, 1000.0, constant]]
        ),
        observable_names=["small", "large", "constant"],
    )
    normalized, factor = dataset.get_normalized()
    np.testing.assert_allclose(normalized.events[:, :2].mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(normalized.events[:, :2].std(axis=0), 1.0)
    np.testing.assert_array_equal(normalized.events[:, 2], 0.0)
    np.testing.assert_allclose((normalized * factor).events, dataset.events)
    # Applying the fitted map to a subset must not refit its moments.
    subset = DataSet(dataset.events[[0, 2]], dataset.observable_names)
    np.testing.assert_array_equal((subset / factor).events, normalized.events[[0, 2]])
    restored = ShiftAndNormalizationFactor.from_mapping(factor.to_mapping())
    np.testing.assert_array_equal((subset / restored).events, (subset / factor).events)


def test_legacy_affine_mapping_retains_its_minus_one_translation():
    factor = ShiftAndNormalizationFactor.from_mapping(
        {"factors": {"x": 2.0}, "offsets": {"x": 3.0}}
    )
    raw = np.array([[3.0], [5.0], [7.0]])
    normalized = factor.normalize_values(raw, ["x"])
    np.testing.assert_array_equal(normalized[:, 0], [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(factor.denormalize_values(normalized, ["x"]), raw)
    np.testing.assert_array_equal(factor.scale_values(raw, ["x"]), raw / 2.0)


def test_legacy_pickled_affine_factor_keeps_its_historical_translation():
    factor = ShiftAndNormalizationFactor({"x": 2.0}, {"x": 3.0})
    # Old trained artifacts pickle this dataclass directly, without a shift.
    factor.__dict__.pop("_shift")
    restored = pickle.loads(pickle.dumps(factor))
    np.testing.assert_array_equal(
        restored.normalize_values(np.array([3.0, 5.0, 7.0]), ["x"]),
        [-1.0, 0.0, 1.0],
    )


@pytest.mark.parametrize("constant", [1e20, -1e20])
def test_repeated_large_constant_normalizes_to_exact_zero(constant):
    dataset = DataSet(np.full((100, 1), constant), ["constant"])
    normalized, factor = dataset.get_normalized()
    np.testing.assert_array_equal(normalized.events, 0.0)
    np.testing.assert_array_equal((normalized * factor).events, dataset.events)


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {ConfigType.DATASET: Path(path)}
        for path in (
            "configs/basic-generated/generated_dataset_config.json",
            "test/configs/dataset/positive_signal.json",
            "test/configs/dataset/unrestricted_signal.json",
            "test/configs/dataset/loaded_extra_observables_resampled.yaml",
            "test/configs/dataset/loaded_extra_observables_replacement.yaml",
            "test/configs/dataset/loaded_extra_observables_identity.yaml",
            "test/configs/dataset/loaded_extra_observables_zero_signal.yaml",
        )
    ],
    indirect=True,
)
def test_control_parameters_are_independent_signal_free_sr_copies(
    function_execution_context,
):
    from dataclasses import asdict

    from data_tools.data_utils import DATASET_REGIONS

    config = function_execution_context.config
    assert len(config.dataset__definitions) == 2
    assert len(config.dataset_parameters) == 4
    for sr_category, cr_category in (
        (DATASET_REGIONS.sr.a, DATASET_REGIONS.cr.a),
        (DATASET_REGIONS.sr.b, DATASET_REGIONS.cr.b),
    ):
        sr = config.get_parameters(sr_category)
        cr = config.get_parameters(cr_category)
        assert cr is config.get_parameters(cr_category)
        assert sr is not cr
        assert type(sr) is type(cr)
        assert not cr.dataset__has_signal
        expected = asdict(sr)
        expected.update(
            category=cr_category,
            dataset__mean_number_of_signal_events=0,
            dataset__number_of_signal_events=0,
            dataset__signal_number_of_events_to_generate=0,
        )
        assert asdict(cr) == expected
        if sr.dataset__signal_generator is not None:
            assert cr.dataset__signal_generator is not sr.dataset__signal_generator
        if isinstance(sr, GeneratedDatasetParameters):
            assert (
                cr.dataset_generated__background_generator
                is not sr.dataset_generated__background_generator
            )
            _, control_signal = cr.dataset__data
            assert control_signal.n_samples == 0
        else:
            assert (
                cr.dataset_loaded__observable_naming
                is not sr.dataset_loaded__observable_naming
            )


@pytest.mark.parametrize(
    "fixture,error,message",
    [
        ("missing_b", ValueError, "exactly two definitions"),
        ("duplicate_a", ValueError, "Duplicate dataset category"),
        ("explicit_cr", ValueError, "must be a or b"),
        ("legacy_four", ValueError, "exactly two definitions"),
        ("missing_name", KeyError, "must contain"),
        ("unknown_type", KeyError, "not defined"),
    ],
)
def test_dataset_schema_is_shared_by_runtime_and_comparison(fixture, error, message):
    import json

    from train.comparison_datasets import dataset_comparison_values

    config = json.loads(
        Path(f"test/configs/dataset/invalid_{fixture}.json").read_text()
    )
    with pytest.raises(error, match=message):
        DatasetConfig(**config).load_dataset_parameters()
    with pytest.raises(error, match=message):
        dataset_comparison_values(config)
