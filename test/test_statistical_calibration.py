from pathlib import Path

import pytest

from test.environment import ConfigType
from train.function_space_config import resolve_dual_role_config
from train.statistical_calibration import (
    CalibrationPolicy,
    calibration_policy,
    effective_test_statistic_degrees_of_freedom,
)


def test_fixed_function_spaces_use_wilks_calibration():
    config = resolve_dual_role_config(
        backend="lfvddp",
        f={
            "family": "orthogonal_polynomial",
            "options": {
                "basis": "legendre",
                "maximum_degree": 2,
                "domain": [[0.0, 1.0]],
            },
        },
        nuisance={"state": "disabled"},
    )

    assert calibration_policy(config) is CalibrationPolicy.WILKS


def test_adaptive_and_nplm_function_spaces_use_empirical_null_calibration():
    adaptive_config = resolve_dual_role_config(
        backend="lfvddp",
        f={"family": "adaptive_neural", "options": {"hidden_layer_nodes": 4}},
        nuisance={"state": "disabled"},
    )
    nplm_config = resolve_dual_role_config(
        backend="nplm",
        f={"family": "adaptive_neural", "options": {"hidden_layer_nodes": 4}},
        nuisance={"state": "disabled"},
    )

    assert calibration_policy(adaptive_config) is CalibrationPolicy.EMPIRICAL_NULL
    assert calibration_policy(nplm_config) is CalibrationPolicy.EMPIRICAL_NULL


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
                    "test/configs/train/issue018_orthogonal_legendre_binned.json"
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
