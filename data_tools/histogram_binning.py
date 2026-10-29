"""Bin edges shared by prediction plots and saved training samples."""

from typing import Iterable

import numpy as np

from data_tools.data_utils import DataSet


DEFAULT_DISPLAY_PERCENTILES = (0.0, 100.0)


def validate_display_percentiles(percentiles: tuple[float, float]) -> None:
    """Require an ordered, finite percentile interval."""
    if len(percentiles) != 2 or not (0.0 <= percentiles[0] < percentiles[1] <= 100.0):
        raise ValueError("Display percentiles must satisfy 0 <= lower < upper <= 100.")


def display_edges_by_observable(
    datasets: Iterable[DataSet],
    observable_names: list[str],
    number_of_bins: int,
    display_percentiles: tuple[float, float] = DEFAULT_DISPLAY_PERCENTILES,
) -> dict[str, np.ndarray]:
    if number_of_bins <= 0:
        raise ValueError(
            f"Expected a positive number of display bins, got {number_of_bins}"
        )

    validate_display_percentiles(display_percentiles)

    datasets = tuple(datasets)
    edges_by_observable = {}
    for observable_name in observable_names:
        values = np.concatenate(
            [
                np.asarray(
                    dataset.slice_along_observable_names(observable_name)
                ).reshape(-1)
                for dataset in datasets
            ]
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                f"Cannot define display bins for {observable_name}: no finite values found."
            )

        minimum, maximum = map(float, np.percentile(values, display_percentiles))
        if minimum == maximum:
            padding = max(abs(minimum) * 0.05, 0.5)
            minimum -= padding
            maximum += padding
        edges_by_observable[observable_name] = np.linspace(
            minimum, maximum, number_of_bins + 1
        )

    return edges_by_observable
