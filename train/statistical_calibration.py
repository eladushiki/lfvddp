"""Wilks-calibration utilities derived from configured function spaces."""

from __future__ import annotations

from neural_networks.function_spaces import create_function_space
from train.function_space_config import (
    FunctionSpaceRole,
)
from train.train_config import TrainConfig


def effective_test_statistic_degrees_of_freedom(
    config: TrainConfig,
) -> int | None:
    """Return the configured signal hypothesis-space dimension.

    The observed event count constrains a constant direction, while fixed
    family implementations own any further structural dependencies.  Neither
    depends on the particular Monte Carlo sample used in a run.
    """

    resolved_config = config.train__function_space_config
    function_space = create_function_space(
        FunctionSpaceRole.F,
        resolved_config.f,
        output_dimension=config.train__nn_output_dimension,
    )
    return function_space.statistical_degrees_of_freedom()


__all__ = [
    "effective_test_statistic_degrees_of_freedom",
]
