from pathlib import Path
import numpy as np
import pytest

from test.environment import ConfigType
from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetConfig

@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        ConfigType.TRAIN.value: Path(
            "test/configs/train/issue018_adaptive_neural_nuisance.json"
        ),
        ConfigType.DETECTOR.value: Path(
            "test/configs/detector/basic_1D_detector_config.json"
        ),
    }],
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
            np.testing.assert_allclose(np.max(normalized.events), 1, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(np.min(normalized.events), -1, rtol=1e-9, atol=1e-9)

    np.testing.assert_array_almost_equal((normalized_A * norm_factor_A).events, A.events)
    np.testing.assert_array_almost_equal((normalized_B * norm_factor_B).events, B.events)


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
    np.testing.assert_allclose(normalized.events[:, 0], [-1.0, 0.0, 1.0])
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

    assert normalization_factor.get_offset("x") == 0.0
    assert normalization_factor.get_factor("x") == 5.0
    expected = normalization_factor.normalize_values(
        np.concatenate(
            [raw_datasets[category].events for category in DataBatch.REQUIRED_DATASET_CATEGORIES]
        ),
        ("x",),
    )
    actual = np.concatenate([dataset.events for dataset, _ in normalized])
    np.testing.assert_allclose(actual, expected)
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
    }
