"""Chi-square diagnostic reference counts derived from configured function spaces."""

from __future__ import annotations

from neural_networks.function_spaces import analytic_degrees_of_freedom
from train.train_config import TrainConfig


def effective_test_statistic_degrees_of_freedom(
    config: TrainConfig,
) -> int:
    """Return the configured signal-space count for chi-square diagnostics.

    The observed event count constrains a constant direction, while fixed
    family implementations own any further structural dependencies.  Neither
    depends on the particular Monte Carlo sample used in a run. Adaptive
    LFVDDP and NPLM use the raw signal-network parameter count. These
    diagnostic counts do not calibrate multi-run significance.
    """

    resolved_config = config.train__function_space_config
    degrees_of_freedom = analytic_degrees_of_freedom(resolved_config.f)
    if degrees_of_freedom is None or degrees_of_freedom <= 0:
        raise ValueError(
            "A diagnostic chi-square reference requires positive degrees of freedom."
        )
    return degrees_of_freedom


__all__ = [
    "effective_test_statistic_degrees_of_freedom",
]
