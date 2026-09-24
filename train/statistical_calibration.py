"""Wilks-calibration utilities derived from configured function spaces."""

from __future__ import annotations

from neural_networks.function_spaces import analytic_degrees_of_freedom
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
    return analytic_degrees_of_freedom(resolved_config.f)


__all__ = [
    "effective_test_statistic_degrees_of_freedom",
]
