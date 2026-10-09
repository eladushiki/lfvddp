"""Adapt the configured nominal detector efficiency to PDF coordinates."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetParameters
from data_tools.detector.detector_config import DetectorConfig
from data_tools.detector.detector_effect import DetectorEffect

if TYPE_CHECKING:
    from frame.context.execution_context import ExecutionContext


def generated_detector_efficiency(
    context: "ExecutionContext | DetectorConfig", dataset_parameters: DatasetParameters
) -> Callable | None:
    """Return the signal dataset's nominal acceptance without sampling events.

    Perfect detectors need no adapter. Efficiency uncertainty and measurement
    error are not part of this efficiency-only analytic benchmark.
    """
    config = context if isinstance(context, DetectorConfig) else context.config
    if not any(
        getattr(config, field, "")
        for field in ("detector__effect_a_efficiency", "detector__effect_b_efficiency")
    ):
        return None

    detector = DetectorEffect(config)
    detector.detection_parameters = dataset_parameters

    def efficiency(coordinates):
        points = np.asarray(coordinates)
        scalar_point = points.ndim < 2
        dataset = DataSet(np.atleast_2d(points)).filter_observable_names(
            list(detector.observable_names)
        )
        values = np.clip(detector.efficiency_values(dataset, nominal=True), 0, 1)
        return values.item() if scalar_point else values

    return efficiency
