"""Wilks-calibration utilities derived from configured function spaces."""

from __future__ import annotations

from frame.value_enum import ValueEnum
from neural_networks.function_spaces import create_function_space
from train.function_space_config import (
    FunctionSpaceFamily,
    FunctionSpaceRole,
    ResolvedFunctionSpaceConfig,
    TrainingBackend,
)
from train.train_config import TrainConfig


class CalibrationPolicy(ValueEnum):
    """Null-calibration choices selected by the configured function spaces."""

    WILKS = "wilks"
    EMPIRICAL_NULL = "empirical-null"


def calibration_policy(
    config: ResolvedFunctionSpaceConfig,
) -> CalibrationPolicy:
    """Select analytic or empirical calibration from the resolved configuration."""

    uses_adaptive_space = any(
        spec.family is FunctionSpaceFamily.ADAPTIVE_NEURAL
        for spec in (config.f, config.nuisance)
        if spec.enabled
    )
    if config.backend is TrainingBackend.NPLM or uses_adaptive_space:
        return CalibrationPolicy.EMPIRICAL_NULL
    return CalibrationPolicy.WILKS


def effective_test_statistic_degrees_of_freedom(
    config: TrainConfig,
) -> int | None:
    """Return the configured signal hypothesis-space dimension.

    The observed event count constrains a constant direction, while fixed
    family implementations own any further structural dependencies.  Neither
    depends on the particular Monte Carlo sample used in a run.
    """

    resolved_config = config.train__function_space_config
    if calibration_policy(resolved_config) is CalibrationPolicy.EMPIRICAL_NULL:
        return None
    function_space = create_function_space(
        FunctionSpaceRole.F,
        resolved_config.f,
        output_dimension=config.train__nn_output_dimension,
    )
    return function_space.statistical_degrees_of_freedom()


__all__ = [
    "CalibrationPolicy",
    "calibration_policy",
    "effective_test_statistic_degrees_of_freedom",
]
