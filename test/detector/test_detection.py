from pathlib import Path
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
    A_affected = detector_effect.affect_and_compensate(A, A_params, False)

    # Expect data to remain unchanged
    assert (A.events == A_affected.events).all()

@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/detector/configs/detector_affected_basic_ds.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_2D_detector_config.json"),
    },{
        ConfigType.DATASET.value: Path("test/detector/configs/detector_affected_basic_ds_2.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_2D_detector_config.json"),
    }],
    indirect=True,
)
def test_detection_effect(
    function_execution_context,
    data_generation,
    detector_effect,
):
    A, A_params = data_generation[DataSet.DataSetCategory.A_SR]
    A_affected = detector_effect.affect_and_compensate(A, A_params, False)

    # Affected dataset should differ from the original one
    # By missing events:
    assert not (A.n_samples == A_affected.n_samples)

    # And by errored observables:
    for i in range(min(100, A_affected.n_samples)):
        event = A.events[i]
        assert not (event in A_affected.events[:100])
