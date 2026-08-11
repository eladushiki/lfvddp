from pathlib import Path

import pytest

from test.environment import ConfigType
from train.train_utils import (
    model_degrees_of_freedom,
    statistic_degrees_of_freedom,
)


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_1D_train_config_without_nuisance.json"
            ),
        },
        {
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_1D_train_config_with_nuisance.json"
            ),
        },
    ],
    indirect=True,
)
def test_test_statistic_degrees_are_the_model_degree_difference(
    function_execution_context,
):
    config = function_execution_context.config

    numerator_degrees = model_degrees_of_freedom(config, is_numerator=True)
    denominator_degrees = model_degrees_of_freedom(config, is_numerator=False)

    assert statistic_degrees_of_freedom(config) == (
        numerator_degrees - denominator_degrees
    )
    assert statistic_degrees_of_freedom(config) == (
        config.train__nn_significant_degrees_of_freedom + 1
    )
