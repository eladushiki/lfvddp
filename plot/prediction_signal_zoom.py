"""Locate injected Gaussian signals for two-dimensional prediction insets."""

import numpy as np

from data_tools.data_utils import DataSet
from data_tools.dataset_config import (
    DatasetConfig,
    DatasetWithGeneratedSignalParameters,
)
from data_tools.event_generation.distribution import GeneratorSpec

_SIGNAL_ZOOM_STANDARD_DEVIATIONS = 1.0


def _gaussian_moments(
    spec: GeneratorSpec, dimensions: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if spec.function == "gaussian_signal":
        return (
            np.full(dimensions, spec.arguments["location"]),
            np.full(dimensions, spec.arguments["gaussian_signal_sigma"] ** 2),
        )
    if spec.function == "multivariate_gaussian_signal":
        return (
            np.asarray(spec.arguments["mean"]),
            np.diag(spec.arguments["covariance"]),
        )
    return None


def prediction_signal_zoom_limits(
    config: DatasetConfig,
    configured_observables: list[str],
    selected_observables: list[str],
) -> np.ndarray | None:
    """Return common mean ± 1 marginal-sigma bounds, ordered by plotted axes.

    Multiple Gaussian injections share the enclosing window. Higher-dimensional
    projections and signals nonresonant in either displayed coordinate omit it.
    """
    if len(configured_observables) != 2:
        return None
    windows = []
    for parameters in config.dataset_parameters:
        if not isinstance(parameters, DatasetWithGeneratedSignalParameters):
            continue
        if not parameters.dataset__has_signal:
            continue
        dimensions = parameters.dataset__signal_number_of_dimensions
        selection = parameters.dataset__signal_generator
        if isinstance(selection, GeneratorSpec):
            moments = _gaussian_moments(selection, dimensions)
        elif isinstance(selection, list):
            components = [_gaussian_moments(spec, 1) for spec in selection]
            moments = (
                tuple(np.concatenate(values) for values in zip(*components))
                if all(component is not None for component in components)
                else None
            )
        else:
            moments = None
        if moments is None:
            continue
        mean, variance = moments
        if np.any(variance <= 0):
            continue
        names = parameters.dataset__signal_observable_names
        if names is None:
            names = DataSet(np.empty((0, dimensions))).observable_names
        if not set(selected_observables).issubset(names):
            continue
        indices = [list(names).index(name) for name in selected_observables]
        half_width = _SIGNAL_ZOOM_STANDARD_DEVIATIONS * np.sqrt(variance[indices])
        center = mean[indices]
        windows.append(np.column_stack((center - half_width, center + half_width)))
    if not windows:
        return None
    return np.column_stack(
        (np.min(windows, axis=0)[:, 0], np.max(windows, axis=0)[:, 1])
    )
