from pathlib import Path
import pytest
from test.environment import ConfigType


def test_context_creation(session_execution_context):
    pass


@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_1D_detector_config.json"),
        ConfigType.TRAIN.value: Path("test/configs/train/short_1D_train_config_without_nuisance.json"),
    }],
    indirect=True,
)
def test_custom_context_creation(function_execution_context):
    assert function_execution_context.config.detector__number_of_dimensions == 1
