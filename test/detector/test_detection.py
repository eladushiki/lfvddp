from pathlib import Path

import numpy as np
import pytest
from test.environment import ConfigType
from data_tools.data_utils import DataSet


@pytest.mark.parametrize(
    "function_execution_context",
    [{  # basic process
        ConfigType.DATASET.value: Path("test/detector/configs/detector_perfect_basic_ds.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_2D_detector_config.json"),
    }],
    indirect=True,
)
def test_detection_basic(
    function_execution_context,
    data_generation,
    detector_effect,
):
    A, A_params = data_generation[DataSet.DataSetCategory.A_SR]
    A_affected = detector_effect.affect_dataset(A, A_params)

    # Expect data to remain unchanged
    assert (A.events == A_affected.events).all()
    np.testing.assert_array_equal(
        detector_effect.efficiency_values(A),
        np.ones(A.n_samples),
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/detector/configs/detector_affected_basic_ds.json"),
        ConfigType.DETECTOR.value: Path("test/detector/configs/detector_affected_basic_detector_config.json"),
    },{
        ConfigType.DATASET.value: Path("test/detector/configs/detector_affected_basic_ds_2.json"),
        ConfigType.DETECTOR.value: Path("test/detector/configs/detector_affected_basic_detector_config_2.json"),
    }],
    indirect=True,
)
def test_detection_effect(
    function_execution_context,
    data_generation,
    detector_effect,
):
    A, A_params = data_generation[DataSet.DataSetCategory.A_SR]
    A_affected = detector_effect.affect_dataset(A, A_params)

    # Affected dataset should differ from the original one
    # By missing events:
    assert not (A.n_samples == A_affected.n_samples)

    # And by errored observables:
    for i in range(min(100, A_affected.n_samples)):
        event = A.events[i]
        assert not (event in A_affected.events[:100])


@pytest.mark.parametrize(
    "function_execution_context",
    [{}],
    indirect=True,
)
def test_detector_exposes_its_canonical_observable_names(detector_effect):
    """Plotting must use names owned by the detector, not stale plot config names."""
    expected_names = tuple(
        detector_effect._context.config.detector__detect_observable_names
    )
    assert detector_effect.observable_names == expected_names
    edges, centers = detector_effect.get_observable_bins(expected_names[0])
    assert len(edges) == 11
    assert len(centers) == 10
