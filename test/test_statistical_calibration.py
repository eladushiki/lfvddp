from pathlib import Path

import pytest

from test.environment import ConfigType
from train.statistical_calibration import effective_test_statistic_degrees_of_freedom


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            {
                ConfigType.DATASET: Path(
                    "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
                ),
                ConfigType.DETECTOR: Path(
                    "test/configs/detector/basic_1D_detector_config.json"
                ),
                ConfigType.TRAIN: Path(
                    "test/configs/train/orthogonal_legendre_binned.json"
                ),
            },
            id="orthogonal-polynomial",
        ),
    ],
    indirect=True,
)
def test_hypothesis_dof_comes_from_configured_function_space(
    function_execution_context,
):
    assert (
        effective_test_statistic_degrees_of_freedom(function_execution_context.config)
        == 3
    )
