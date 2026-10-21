"""All data sources must enter regional pools in the detector observable space."""

from pathlib import Path

import numpy as np
import pytest

from data_tools.data_utils import DATASET_REGIONS
from test.environment import ConfigType


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET: Path(
                f"test/configs/dataset/loaded_extra_observables_{mode}.yaml"
            ),
            ConfigType.DETECTOR: Path(f"test/configs/detector/{detector_fixture}.json"),
            ConfigType.TRAIN: Path(
                f"test/configs/train/short_{dimension}D_train_config_without_nuisance.json"
            ),
        }
        for mode in ("identity", "resampled", "replacement", "zero_signal")
        for dimension, detector_fixture in (
            (1, "basic_1D_detector_config"),
            (2, "basic_2D_detector_config"),
            (2, "reordered_2D_detector_config"),
        )
    ],
    indirect=True,
)
def test_loaded_pair_projection_preserves_columns_counts_and_source_depletion(
    tmp_path,
    monkeypatch,
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    loaded_extra_observable_source,
):
    source = loaded_extra_observable_source
    config = function_execution_context.config
    names = list(config.detector__detect_observable_names)
    signal_parameters = config.get_parameters(DATASET_REGIONS.sr.a)
    replacement = signal_parameters.dataset_loaded__resample_is_replacement
    signal_count = signal_parameters.dataset__number_of_signal_events

    with monkeypatch.context() as working_directory:
        working_directory.chdir(tmp_path)
        for batch_index in range(2):
            batch = isolated_data_generation.get_batch()
            for dataset, parameters in batch:
                assert dataset.observable_names == names
                assert dataset.n_samples == (
                    parameters.dataset__number_of_background_events
                    + parameters.dataset__number_of_signal_events
                )
            for region, injected in (
                (DATASET_REGIONS.sr, signal_count),
                (DATASET_REGIONS.cr, 0),
            ):
                pooled = np.concatenate(
                    [batch.datasets[region.a].events, batch.datasets[region.b].events]
                )
                background = pooled[pooled[:, 0] >= 10]
                expected = source[3 * batch_index : 3 * (batch_index + 1)][
                    :, [int(name.removeprefix("param_")) for name in names]
                ]
                if replacement:
                    assert set(map(tuple, background)) <= set(map(tuple, expected))
                else:
                    np.testing.assert_array_equal(
                        background[np.argsort(background[:, 0])],
                        np.repeat(expected, 2, axis=0),
                    )
                    assert len(pooled) - len(background) == injected
            detected = detector_effect.affect_batch(batch)
            assert all(dataset.observable_names == names for dataset, _ in detected)
            normalized, _ = batch.get_normalized()
            assert all(np.isfinite(dataset.events).all() for dataset, _ in normalized)

        with pytest.raises(ValueError, match="only 0 samples"):
            isolated_data_generation.get_batch()


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET: Path(
                "test/configs/dataset/loaded_extra_observables_resampled.yaml"
            ),
            ConfigType.DETECTOR: Path(
                "test/configs/detector/basic_2D_detector_config.json"
            ),
            ConfigType.TRAIN: Path(
                "test/configs/train/short_2D_train_config_without_nuisance.json"
            ),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "category",
    [
        DATASET_REGIONS.sr.a,
        DATASET_REGIONS.sr.b,
        DATASET_REGIONS.cr.a,
        DATASET_REGIONS.cr.b,
    ],
)
def test_category_access_uses_projected_regional_pool(
    tmp_path,
    monkeypatch,
    isolated_data_generation,
    loaded_extra_observable_source,
    function_execution_context,
    category,
):
    with monkeypatch.context() as working_directory:
        working_directory.chdir(tmp_path)
        dataset, parameters = isolated_data_generation[category]
    assert dataset.observable_names == list(
        function_execution_context.config.detector__detect_observable_names
    )
    assert dataset.category == category
    assert (
        dataset.n_samples
        == parameters.dataset__number_of_background_events
        + parameters.dataset__number_of_signal_events
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET: Path(
                "test/configs/dataset/generated_extra_observables.yaml"
            ),
            ConfigType.DETECTOR: Path(f"test/configs/detector/{detector_fixture}.json"),
            ConfigType.TRAIN: Path(
                f"test/configs/train/short_{dimension}D_train_config_without_nuisance.json"
            ),
        }
        for dimension, detector_fixture in (
            (1, "basic_1D_detector_config"),
            (2, "basic_2D_detector_config"),
            (2, "reordered_2D_detector_config"),
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("entry_point", ["batch", "category"])
def test_generated_source_uses_the_same_ordered_projection(
    function_execution_context,
    isolated_data_generation,
    entry_point,
):
    config = function_execution_context.config
    category = DATASET_REGIONS.sr.a
    parameters = config.get_parameters(category)
    random_state = np.random.get_state()
    background, signal = parameters.dataset__data
    raw_events = np.concatenate((background.events, signal.events))
    np.random.set_state(random_state)

    if entry_point == "batch":
        batch = isolated_data_generation.get_batch()
        assert all(
            dataset.observable_names == list(config.detector__detect_observable_names)
            for dataset, _ in batch
        )
        dataset = batch.datasets[category]
    else:
        dataset, _ = isolated_data_generation[category]
    names = list(config.detector__detect_observable_names)
    indices = [background.observable_names.index(name) for name in names]
    assert raw_events.shape == (5, 4)
    assert dataset.observable_names == names
    np.testing.assert_array_equal(dataset.events, raw_events[:, indices])
