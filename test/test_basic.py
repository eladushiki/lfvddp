from pathlib import Path

import pytest

from test.environment import ConfigType
from train.train_config import TrainConfig


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


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"train__optimizer": "sgd"}, "Unsupported optimizer"),
        ({"train__learning_rate": 0}, "learning rate must be positive"),
    ],
)
def test_train_config_rejects_invalid_optimizer_settings(overrides, message):
    parameters = {
        "train__epochs": 1,
        "train__number_of_epochs_for_checkpoint": 1,
        "train__nn_input_dimension": 1,
        "train__nn_inner_layer_nodes": 4,
        "train__nn_output_dimension": 1,
    }
    parameters.update(overrides)

    with pytest.raises(ValueError, match=message):
        TrainConfig(**parameters)
