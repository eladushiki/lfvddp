"""Flat, comparable views of multidimensional prediction-process results."""

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from data_tools.data_utils import DataSet
from neural_networks.utils import prediction_to_sample_ndf_hypothesis_weights


@dataclass(frozen=True)
class RegionDistributions:
    """Observed and weighted-prediction histograms, with counting variances."""

    maps: list[tuple[str, np.ndarray]]
    difference: np.ndarray


def region_distributions(
    sample_a: DataSet,
    sample_b: DataSet,
    predictions: tuple[np.ndarray, np.ndarray],
    prediction_labels: tuple[str, str],
    bins: list[np.ndarray],
    observables: list[str],
    normalize: bool,
) -> RegionDistributions:
    """Bin the same events and model weights as the line/surface plotters."""
    background = sample_a + sample_b

    def histogram(dataset: DataSet, weights=None) -> np.ndarray:
        coordinates = dataset.slice_along_observable_names(observables)
        return np.histogram2d(*coordinates.T, bins=bins, weights=weights)[0]

    samples = (sample_a, sample_b)
    counts = [histogram(sample) for sample in samples]
    factors = [
        1.0 / max(sample.n_samples, 1) if normalize else 1.0 for sample in samples
    ]
    observed = [count * factor for count, factor in zip(counts, factors)]
    variance = sum(count * factor**2 for count, factor in zip(counts, factors))
    difference = np.divide(
        observed[0] - observed[1],
        np.sqrt(variance),
        out=np.full_like(variance, np.nan),
        where=variance > 0,
    )
    predicted = []
    for sample, prediction, factor in zip(samples, predictions, factors):
        weights = prediction_to_sample_ndf_hypothesis_weights(
            model_prediction=prediction,
            predicted_distribution_size=sample.n_samples,
            reference_ndf_estimation=background,
        )
        predicted.append(
            histogram(background, np.asarray(weights).reshape(-1)) * factor
        )
    pooled = histogram(background)
    if normalize:
        pooled /= max(background.n_samples, 1)
    return RegionDistributions(
        maps=[
            ("A + B (pooled background)", pooled),
            ("A data", observed[0]),
            ("B data", observed[1]),
            (prediction_labels[0], predicted[0]),
            (prediction_labels[1], predicted[1]),
        ],
        difference=difference,
    )


def _finite_values(arrays: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([array[np.isfinite(array)].ravel() for array in arrays])


def _centered_norm(arrays: list[np.ndarray], center: float) -> TwoSlopeNorm:
    values = _finite_values(arrays)
    half_range = max(float(np.max(np.abs(values - center))) if values.size else 0, 0.05)
    return TwoSlopeNorm(
        vmin=center - half_range, vcenter=center, vmax=center + half_range
    )


def _map_axis(
    fig: Figure,
    slot: SubplotSpec,
    label: str,
    column: int,
    row: int,
    shape: tuple[int, int],
    bins: list[np.ndarray],
    observables: list[str],
):
    ax = fig.add_subplot(slot)
    # These identify the data, including in non-debug exports where titles are removed.
    ax.text(
        0.5, 1.025, label, ha="center", va="bottom", transform=ax.transAxes, fontsize=9
    )
    ax.set(xlim=(bins[0][0], bins[0][-1]), ylim=(bins[1][0], bins[1][-1]))
    ax.tick_params(
        labelsize=8,
        top=False,
        right=False,
        labelbottom=row == shape[0] - 1,
        labelleft=column == 0,
    )
    ax.locator_params(nbins=4)
    ax.grid(False)
    if row == shape[0] - 1:
        ax.set_xlabel(observables[0], fontsize=10)
    if column == 0:
        ax.set_ylabel(observables[1], fontsize=10)
    return ax


def _block_grid(fig: Figure, slot: SubplotSpec, heading: str, columns: int):
    grid = slot.subgridspec(
        4, columns, height_ratios=[0.07, 1, 1, 0.06], hspace=0.02, wspace=0.02
    )
    heading_ax = fig.add_subplot(grid[0, :])
    heading_ax.set_axis_off()
    heading_ax.text(0, 0.5, heading, fontsize=13, weight="bold", va="center")
    return grid


def _colorbar(fig: Figure, slot: SubplotSpec, artist, label: str):
    ax = fig.add_subplot(slot)
    bar = fig.colorbar(artist, cax=ax, orientation="horizontal")
    bar.set_label(label, fontsize=9, labelpad=2)
    bar.ax.tick_params(labelsize=8)
    return bar


def render_prediction_heatmaps(
    fig: Figure,
    distributions: tuple[RegionDistributions, RegionDistributions],
    predictions: tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]],
    coordinates: np.ndarray,
    bins: list[np.ndarray],
    observables: list[str],
    normalize: bool,
    log_scale: bool,
) -> None:
    """Render SR/CR blocks with shared scales, exterior labels, and no occlusion."""
    outer = fig.add_gridspec(2, 2, hspace=0.035, wspace=0.04)
    density_values = _finite_values(
        [values for region in distributions for _, values in region.maps]
    )
    positive = density_values[density_values > 0]
    maximum = float(np.max(positive)) if positive.size else 1.0
    if log_scale:
        minimum = float(np.min(positive)) if positive.size else 0.1
        density_norm = LogNorm(
            vmin=minimum, vmax=maximum if maximum > minimum else minimum * 10
        )
    else:
        density_norm = Normalize(vmin=0, vmax=maximum)
    difference_norm = _centered_norm([region.difference for region in distributions], 0)
    prediction_norm = _centered_norm(
        [values for region in predictions for _, values in region], 1
    )
    density_label = "Probability per bin" if normalize else "Events per bin"
    x_values, y_values = (np.unique(coordinates[:, index]) for index in range(2))
    for column, (name, region, model_maps) in enumerate(
        zip(("SR", "CR"), distributions, predictions)
    ):
        grid = _block_grid(
            fig, outer[0, column], f"{name} distributions and fitted densities", 3
        )
        distribution_maps = [(label, values, False) for label, values in region.maps]
        distribution_maps.insert(3, ("A/B difference", region.difference, True))
        for index, (label, values, is_difference) in enumerate(distribution_maps):
            row, col = divmod(index, 3)
            ax = _map_axis(
                fig, grid[row + 1, col], label, col, row, (2, 3), bins, observables
            )
            artist = ax.pcolormesh(
                *bins,
                np.ma.masked_invalid(values.T),
                cmap="RdBu_r" if is_difference else "viridis",
                norm=difference_norm if is_difference else density_norm,
                shading="flat",
                rasterized=True,
            )
            if is_difference:
                _colorbar(fig, grid[3, 0], artist, "Difference / counting error")
            elif index == 0:
                _colorbar(
                    fig,
                    grid[3, 1:],
                    artist,
                    density_label + (" (log color)" if log_scale else ""),
                )

        grid = _block_grid(
            fig, outer[1, column], f"{name} model factors (neutral = 1)", 2
        )
        for index, (label, values) in enumerate(model_maps):
            row, col = divmod(index, 2)
            ax = _map_axis(
                fig, grid[row + 1, col], label, col, row, (2, 2), bins, observables
            )
            artist = ax.pcolormesh(
                x_values,
                y_values,
                np.ma.masked_invalid(values.reshape(len(x_values), len(y_values)).T),
                cmap="RdBu_r",
                norm=prediction_norm,
                shading="nearest",
                rasterized=True,
            )
        _colorbar(fig, grid[3, :], artist, "Model factor (shared SR/CR scale)")
