"""Wilks-calibration utilities derived from configured function spaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from frame.value_enum import ValueEnum
from neural_networks.function_spaces.projected_rank import compute_rank_for_backend
from train.function_space_config import (
    FunctionSpaceFamily,
    ResolvedFunctionSpaceConfig,
    TrainingBackend,
)

if TYPE_CHECKING:
    from data_tools.data_generation import DataBatch
    from neural_networks.differentiating_model import DifferentiatingModel


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
    model: "DifferentiatingModel",
    data: "DataBatch",
) -> int | None:
    """Return the projected fixed-basis rank for one recreated numerator model.

    The rank is evaluated on the detected batch because empty bins and overlaps
    with the fitted nuisance space are properties of that concrete experiment,
    not merely the declared number of basis functions.
    """

    config = model._function_space_config
    if calibration_policy(config) is CalibrationPolicy.EMPIRICAL_NULL:
        return None

    f_design, nuisance_design = model.statistical_design_matrices(data)
    if f_design is None:
        raise RuntimeError(
            "A Wilks-calibrated function-space model did not provide a statistical design."
        )
    rank = compute_rank_for_backend(
        f_design,
        nuisance_design,
        backend=config.backend,
    )
    # A fully projected-out f has no non-degenerate chi-square limit.
    return rank.effective_f_rank or None


__all__ = [
    "CalibrationPolicy",
    "calibration_policy",
    "effective_test_statistic_degrees_of_freedom",
]
