from pathlib import Path

import pytest

from test.environment import ConfigType
from test.function_space_cases import NEURAL_DEPTH_CASES, NEURAL_DEPTH_CONFIGS
from train.statistical_calibration import effective_test_statistic_degrees_of_freedom


@pytest.mark.parametrize(
    ("function_execution_context", "expected_dof"),
    [
        *[
            pytest.param(config, case[3], id=case[0])
            for config, case in zip(NEURAL_DEPTH_CONFIGS, NEURAL_DEPTH_CASES)
        ],
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
            3,
            id="orthogonal-polynomial",
        ),
        *[
            pytest.param(
                {
                    ConfigType.DATASET: Path(
                        f"test/configs/dataset/disjoint_{dimension}D_generated_dataset_config.json"
                    ),
                    ConfigType.DETECTOR: Path(
                        f"test/configs/detector/basic_{dimension}D_detector_config.json"
                    ),
                    ConfigType.TRAIN: Path(f"test/configs/train/{fixture}.json"),
                },
                expected_dof,
                id=fixture,
            )
            for fixture, dimension, expected_dof in [
                ("short_1D_train_config_with_neural_nuisance", 1, 13),
                ("two_dimensional_adaptive_neural_binned", 2, 17),
                ("bin_indicators_binned", 1, 3),
                ("fixed_sigmoid_binned", 1, 2),
                ("gaussian_radial_basis_binned", 1, 2),
                ("cubic_bspline_binned", 1, 6),
                ("orthogonal_chebyshev_binned", 1, 3),
            ]
        ],
    ],
    indirect=["function_execution_context"],
)
def test_hypothesis_dof_comes_from_configured_function_space(
    function_execution_context,
    expected_dof,
):
    assert (
        effective_test_statistic_degrees_of_freedom(function_execution_context.config)
        == expected_dof
    )


def test_nplm_configuration_keeps_its_raw_count_without_importing_backend():
    from frame.command_line.handle_args import create_config_from_paths
    from test.environment import DEFAULT_CONFIG_PATHS

    paths = {
        **DEFAULT_CONFIG_PATHS,
        ConfigType.TRAIN: Path(
            "test/configs/train/short_1D_train_config_without_nuisance_like_nplm.json"
        ),
    }
    config = create_config_from_paths(list(paths.values()))
    assert config.train__adaptive_architecture == [1, 4, 1]
    assert effective_test_statistic_degrees_of_freedom(config) == 13
