import json

import numpy as np
import pytest
from pathlib import Path

from data_tools.dataset_config import DatasetConfig
from data_tools.dataset_pair import (
    RegionalDataPair,
    ShuffledDatasetPairSplitPolicy,
)
from test.environment import ConfigType
from data_tools.data_utils import DataSet


@pytest.mark.parametrize(
        "function_execution_context",
        [{
            ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        }],
        indirect=True,
)
def test_illegal_data_request(
        data_generation,
):
    with pytest.raises(KeyError):
        _, _ = data_generation[DataSet.DataSetCategory.SR]

@pytest.mark.parametrize(
        "function_execution_context",
        [{
            ConfigType.DATASET.value: Path("test/data_generation/configs/dataset/small_exact_sized_loaded_dataset_config.json"),
        }],
        indirect=True,
)
def test_resampling_exhaustion(
        tmp_path,
        monkeypatch,
        isolated_data_generation,
):
    np.save(tmp_path / "events.npy", np.arange(2, dtype=float).reshape(-1, 1))

    with monkeypatch.context() as temporary_working_directory:
        temporary_working_directory.chdir(tmp_path)
        isolated_data_generation.get_batch()

        # A second full A/B-pair generation cannot draw any background event.
        with pytest.raises(ValueError, match="has only 0 samples left"):
            isolated_data_generation.get_batch()


def test_shuffled_regional_pair_preserves_pool_and_can_move_signal(monkeypatch):
    categories = DataSet.DataSetCategory
    first = DataSet(
        np.array([[0.0], [100.0]]),
        observable_names=["x"],
        category=categories.A_SR,
    )
    second = DataSet(
        np.array([[10.0], [11.0], [12.0]]),
        observable_names=["x"],
        category=categories.B_SR,
    )
    monkeypatch.setattr(
        np.random,
        "permutation",
        lambda size: np.array([0, 2, 3, 4, 1]),
    )

    pair = RegionalDataPair(
        a=first,
        a_parameters=None,
        b=second,
        b_parameters=None,
        split_policy=ShuffledDatasetPairSplitPolicy(replacement=False),
    ).finalized()
    first_result = pair.a
    second_result = pair.b

    assert first_result.category == categories.A_SR
    assert second_result.category == categories.B_SR
    assert first_result.n_samples == first.n_samples
    assert second_result.n_samples == second.n_samples
    np.testing.assert_array_equal(
        np.sort(np.concatenate((first_result.events, second_result.events), axis=0), axis=0),
        np.sort(np.concatenate((first.events, second.events), axis=0), axis=0),
    )
    assert 100.0 in second_result.events


@pytest.mark.parametrize(
        "function_execution_context",
        [{
            ConfigType.DATASET.value: Path(
                "test/data_generation/configs/dataset/resampled_pairs_dataset_config.json"
            ),
        }],
        indirect=True,
)
def test_loaded_resampling_shuffles_complete_regional_pairs(
        tmp_path,
        monkeypatch,
        isolated_data_generation,
):
    for filename, values in {
        "sr_a.npy": np.arange(6, dtype=float),
        "sr_b.npy": np.arange(10, 16, dtype=float),
        "cr_a.npy": np.arange(100, 106, dtype=float),
        "cr_b.npy": np.arange(200, 206, dtype=float),
    }.items():
        np.save(tmp_path / filename, values.reshape(-1, 1))

    monkeypatch.setattr(np.random, "permutation", lambda size: np.arange(size)[::-1])
    with monkeypatch.context() as temporary_working_directory:
        temporary_working_directory.chdir(tmp_path)
        batch = isolated_data_generation.get_batch()

    a_sr = batch.datasets[DataSet.DataSetCategory.A_SR]
    b_sr = batch.datasets[DataSet.DataSetCategory.B_SR]
    a_cr = batch.datasets[DataSet.DataSetCategory.A_CR]
    b_cr = batch.datasets[DataSet.DataSetCategory.B_CR]

    assert a_sr.events.shape == (4, 1)
    assert b_sr.events.shape == (3, 1)
    assert a_cr.events.shape == (2, 1)
    assert b_cr.events.shape == (3, 1)
    sr_events = np.concatenate((a_sr.events, b_sr.events), axis=0)
    cr_events = np.concatenate((a_cr.events, b_cr.events), axis=0)
    assert np.all((sr_events < 20.0) | (sr_events > 500.0))
    assert np.all((cr_events >= 100.0) & (cr_events < 206.0))
    assert np.any(b_sr.events > 500.0)


@pytest.mark.parametrize(
    "config_filename",
    [
        "mismatched_resampling_pair_dataset_config.json",
        "mismatched_resampling_enabled_pair_dataset_config.json",
    ],
)
def test_dataset_config_rejects_mismatched_regional_resampling_policy(
    config_filename,
):
    config_path = Path("test/data_generation/configs/dataset/") / config_filename
    config = DatasetConfig(**json.loads(config_path.read_text()))

    with pytest.raises(ValueError, match="matching regional resampling settings"):
        config.load_dataset_parameters()
