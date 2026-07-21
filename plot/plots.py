from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import patches
from matplotlib.figure import Figure
from scipy.stats import chi2

from data_tools.data_utils import DataSet
from data_tools.dataset_config import (
    DatasetConfig,
    DatasetParameters,
    GeneratedDatasetParameters,
)
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_continuous,
    calc_median_t_significance_relative_to_background,
    calc_t_significance_by_gaussian_fit_percentile,
    calc_t_significance_relative_to_background,
)
from frame.aggregate import ResultAggregator, utils__get_signal_dataset_parameters
from frame.context.execution_context import ExecutionContext
from frame.file_structure import CONTEXT_FILE_NAME
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import prediction_to_sample_ndf_hypothesis_weights
from plot.carpenter import Carpenter
from plot.plot_utils import (
    HandlerCircle,
    HandlerRect,
    utils__add_subplot_sliced,
    utils__datset_histogram_sliced,
    utils__flatten_histogram_values,
    utils__model_prediction_values,
    utils__plot_model_predictions_sliced,
    utils__plot_region_histogram_meshes_2d,
    utils__plot_region_histograms_sliced,
    utils__plot_weighted_histogram_predictions_sliced,
    utils__prediction_process_observables,
    utils__project_prediction_values_sliced,
    utils__remove_eta_from_prediction_values,
    utils__synchronize_output_axis_limits,
)
from plot.plotting_config import PlottingConfig
from train.model_trainer import TrainLauncher
from train.train_config import TrainConfig
from train.train_utils import model_degrees_of_freedom

# DEVELOPER NOTE: Each function here can ba called from "PlottingConfig" BY NAME.
# Implement any new plot function here, and you will be able to call it automatically.
# This being said, the format for implementation has to be:
#
# def <name from plot_config.json>(context: ExecutionContext, <instructions from plot_config.json>) -> matplotlib.figure.Figure:
#    ...
#
# Should not save the figure by itself!!! It is done in a well documented way in the calling function.


def t_train_percentile_progression_plot(
    context: ExecutionContext,
):
    """
    Plot numerator, denominator, and derived-t percentile progression for each
    sample over the toy runs.
    """
    if not isinstance(config := context.config, PlottingConfig):
        raise ValueError(
            f"Expected context.config to be of type {PlottingConfig}, got {type(config)}"
        )

    # Training results aggregation
    agg = ResultAggregator(Path(config.plot__target_run_parent_directory))
    all_history_values = agg.all_history_values
    epochs = agg.all_epochs

    # Framing
    c = Carpenter(context)
    fig = c.figure()
    sample_names = sorted(all_history_values)
    axes = fig.subplots(len(sample_names), 1, squeeze=False, sharex=True)

    quantiles = [2.5, 25, 50, 75, 97.5]
    colors = ["violet", "hotpink", "mediumvioletred", "mediumorchid", "darkviolet"]
    legend_handles = []
    for row, sample_name in enumerate(sample_names):
        ax = axes[row, 0]
        values = all_history_values[sample_name][HistoryKeys.T.value]
        percentiles = np.nanpercentile(values, quantiles, axis=0)
        for quantile, percentile, color in zip(quantiles, percentiles, colors):
            (line,) = ax.plot(
                epochs,
                percentile,
                linewidth=2,
                color=color,
                label=f"{quantile}% quantile",
            )
            if row == 0:
                legend_handles.append(line)
        ax.set_title(rf"{sample_name}: $t=-2\,N+2\,D$")
        ax.set_ylabel(HistoryKeys.T.value)
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
        if row == len(sample_names) - 1:
            ax.set_xlabel("Training epochs")

    fig.suptitle("Training percentile progression", fontsize=24)
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        frameon=False,
        loc="upper center",
        ncol=len(quantiles),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    return fig


def t_distribution_plot(
    context: ExecutionContext,
    number_of_bins: int,
    cut_non_converged: bool = True,
) -> Figure:
    """
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution.
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    """
    if not isinstance(config := context.config, PlottingConfig):
        raise ValueError(
            f"Expected context.config to be of type {PlottingConfig}, got {type(config)}"
        )
    if not isinstance(config, TrainConfig):
        raise ValueError(
            f"Expected context.config to be of type {TrainConfig}, got {type(config)}"
        )
    if not isinstance(config, DetectorConfig):
        raise ValueError(
            f"Expected context.config to be of type {DetectorConfig}, got {type(config)}"
        )
    style = config.plot__figure_styling["plot"]

    # Figure
    c = Carpenter(context)
    fig = c.figure()
    ax = fig.add_subplot(111)

    agg = ResultAggregator(Path(config.plot__target_run_parent_directory))
    t = agg.all_t_values

    # Convergence statistics
    fifth_percentile = np.percentile(t, 5)
    critical_mass_t = t >= fifth_percentile
    distribution_std = np.std(t[critical_mass_t])
    distribution_mean = np.mean(t[critical_mass_t])
    n_std = 6
    did_not_converge = t < (distribution_mean - n_std * distribution_std)
    if cut_non_converged:
        t = t[~did_not_converge]

    # Limits
    chi2_begin = 0
    chi2_end = chi2.ppf(0.9999, chi2_dof := model_degrees_of_freedom(config))
    xmin = min(0.0, float(np.min(t)))
    xmax = max(0.0, float(np.max(t)))
    if xmin == xmax:
        xmax = xmin + max(1.0, abs(xmin) * 0.1)

    # plot distribution histogram
    histogram_bins = np.linspace(xmin, xmax, number_of_bins + 1)
    histogram_bin_width = (xmax - xmin) / number_of_bins
    histogram_bin_centers = 0.5 * (histogram_bins[1:] + histogram_bins[:-1])
    label = (
        f"median: {str(np.around(np.median(t), 2))} \n"
        f"mean: {str(np.around(distribution_mean, 2))} \n"
        f"std: {str(np.around(distribution_std, 2))}"
    )

    invalid_t_num = did_not_converge.sum() + agg.nan_t_values
    if invalid_t_num > 0:
        label += f"\ndid not converge: {invalid_t_num / t.size * 100:.2f}%"

    h, _, _ = ax.hist(
        t,
        weights=np.full(t.shape, 1.0 / (t.size * histogram_bin_width)),
        color=style["histogram_color"],
        ec=style["edge_color"],
        bins=histogram_bins,
        label=label,
    )

    y_error = np.sqrt(h / (t.shape[0] * histogram_bin_width))
    ax.errorbar(
        histogram_bin_centers,
        h,
        yerr=y_error,
        color=style["edge_color"],
        marker="o",
        ls="",
    )

    # plot reference chi2
    chi2_bin_centers = np.linspace(chi2_begin, chi2_end, 1000)

    ax.plot(
        chi2_bin_centers,
        chi2.pdf(chi2_bin_centers, chi2_dof),
        style["chi2_color"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        label=fr"$\chi^{{2}}_{{{chi2_dof}}}$",
    )

    # Legend
    circ = patches.Circle(
        (0, 0), 1, facecolor=style["histogram_color"], edgecolor=style["edge_color"]
    )
    rect1 = patches.Rectangle(
        (0, 0), 1, 1, color=style["chi2_color"], alpha=style["alpha"]
    )

    ax.legend(
        (circ, rect1),
        (label, fr"$\chi^{{2}}_{{{chi2_dof}}}$"),
        handler_map={
            patches.Rectangle: HandlerRect(),
            patches.Circle: HandlerCircle(),
        },
        frameon=False,
    )

    # Texting
    histogram_title = f"Distribution of t values over {len(t)} test runs"
    ax.set_title(histogram_title, fontsize=30, pad=20)
    ax.set_xlabel("t", fontsize=22, labelpad=20)
    ax.set_ylabel("Bin Probability", fontsize=22, labelpad=20)
    ax.set_ylim(0, top=float(np.max(h + y_error)) * 1.05)
    ax.set_xlim(xmin, xmax)
    plt.yticks()
    plt.xticks()

    return fig


def performance_plot(
    context: ExecutionContext,
    background_only_t_values_parent_directory: str,
    signal_t_values_parent_directories: List[str],
):
    """
    Create a plot of the measured significance as a function of
    the ideal z = sqrt(q0) with a given background and signal
    types.

    Data needed to generate the plot:
    - t values distribution for a run with background only.
        contained in a single directory and is used as a
        reference for all signal distributions.
    - A set of t values distributions, each with a different
        injected signal strength. Parameters of each are picked
        from the context file, from the data specification under
        the corresponding signal dataset name BY ORDER.

    The plot__target_run_parent_directory has no use here to
    not cause ambiguity.
    """
    if not isinstance(plot_config := context.config, PlottingConfig):
        raise ValueError(
            f"Expected context.config to be of type {PlottingConfig}, got {type(plot_config)}"
        )

    # Validate background configuration
    ## this has to be a generated type, else the distribution is not well known
    background_context = ExecutionContext.naive_load_from_file(
        Path(background_only_t_values_parent_directory) / CONTEXT_FILE_NAME
    )
    background_config: DatasetConfig = background_context.config
    for background_dataset_name in background_config._dataset__names:
        background_dataset_properties: DatasetParameters = (
            background_config._dataset__parameters(background_dataset_name)
        )
        assert background_dataset_properties.dataset__number_of_signal_events == 0, (
            f"background dataset expected to have only background events, {background_dataset_name} has {background_dataset_properties.dataset__number_of_signal_events} signal events"
        )

    # Gather background data
    background_agg = ResultAggregator(Path(background_only_t_values_parent_directory))
    background_t_dist = background_agg.all_t_values

    # Result lists
    ## The analytic calculation of significance based on input parameters, by eq. (33) in the last paper
    mean_injected_significances = []
    injected_significance_stds = []
    mean_signal_strengths = []

    ## The significance by the observed chance to generate an equal or larger t value had this been a
    ## background only dataset, and confidence bounds
    observed_significances = []
    observed_significances_upper_confidence_bounds = []
    observed_significances_lower_confidence_bounds = []
    observed_significances_by_gaussian_fit = []

    for signal_t_values_dir in signal_t_values_parent_directories:
        # Load corresponding dataset
        signal_context = ExecutionContext.naive_load_from_file(
            Path(signal_t_values_dir) / CONTEXT_FILE_NAME
        )
        signal_dataset_parameters = utils__get_signal_dataset_parameters(signal_context)

        # Gather data
        signal_agg = ResultAggregator(Path(signal_t_values_dir))
        signal_t_dist = signal_agg.all_t_values

        # Calculate the injected significance centers using the mean number of events.
        # Those are before introducting poisson fluctuations.
        if isinstance(signal_dataset_parameters, GeneratedDatasetParameters):
            mean_injected_significances.append(
                calc_injected_t_significance_by_sqrt_q0_continuous(
                    background_pdf=signal_dataset_parameters.dataset_generated__background_pdf,
                    signal_pdf=signal_dataset_parameters.dataset_generated__signal_pdf,
                    n_background_events=signal_dataset_parameters.dataset__mean_number_of_background_events,
                    n_signal_events=signal_dataset_parameters.dataset__mean_number_of_signal_events,
                    upper_limit=max(signal_t_dist.max(), background_t_dist.max()),
                )
            )
            injected_significance_stds.append(
                np.std(signal_agg.all_injected_significances)
            )
        else:
            mean_signal_strengths.append(
                signal_dataset_parameters.dataset__number_of_signal_events
            )
            injected_significance_stds.append(0.0)

        # Calculate observed significance and +-1 sigma confidence interval
        observed_significances.append(
            calc_median_t_significance_relative_to_background(
                background_t_dist,
                signal_t_dist,
            )
        )
        signal_t_dist_std = np.std(signal_t_dist)
        observed_significances_lower_confidence_bounds.append(
            calc_t_significance_relative_to_background(
                np.mean(signal_t_dist) - signal_t_dist_std, background_t_dist
            )
        )
        observed_significances_upper_confidence_bounds.append(
            calc_t_significance_relative_to_background(
                np.mean(signal_t_dist) + signal_t_dist_std, background_t_dist
            )
        )
        observed_significances_by_gaussian_fit.append(
            calc_t_significance_by_gaussian_fit_percentile(
                background_only_distribution=background_t_dist,
                t_value=np.median(signal_t_dist),
            )
        )

    # Sort all results by injected significance
    if not len(mean_injected_significances) == 0:
        sort = np.argsort(np.array(mean_injected_significances))
        mean_injected_significances = np.array(mean_injected_significances)[sort]
        plot_x = mean_injected_significances
        x_label = r"injected $\sqrt{q_0}$"
    else:
        sort = np.argsort(np.array(mean_signal_strengths))
        plot_x = np.array(mean_signal_strengths)[sort]
        x_label = r"mean signal number of events"
    injected_significance_stds = np.array(injected_significance_stds)[sort]
    observed_significances = np.array(observed_significances)[sort]
    observed_significances_lower_confidence_bounds = np.array(
        observed_significances_lower_confidence_bounds
    )[sort]
    observed_significances_upper_confidence_bounds = np.array(
        observed_significances_upper_confidence_bounds
    )[sort]
    observed_significances_by_gaussian_fit = np.array(
        observed_significances_by_gaussian_fit
    )[sort]

    # Framing
    c = Carpenter(context)
    fig = c.figure()
    ax = fig.add_subplot(111)

    # Borders
    graph_border = 1
    clean_y_significances = np.concatenate(
        [
            observed_significances[np.isfinite(observed_significances)],
            observed_significances_lower_confidence_bounds[
                np.isfinite(observed_significances_lower_confidence_bounds)
            ],
            observed_significances_upper_confidence_bounds[
                np.isfinite(observed_significances_upper_confidence_bounds)
            ],
            observed_significances_by_gaussian_fit[
                np.isfinite(observed_significances_by_gaussian_fit)
            ],
        ]
    )

    if not len(mean_injected_significances) == 0:
        min_x = max(min(mean_injected_significances) - graph_border, 0)
        max_x = max(mean_injected_significances) + graph_border
    else:
        min_x = max(min(mean_signal_strengths) - graph_border, 0)
        max_x = max(mean_signal_strengths) + graph_border
    min_y = max(min(clean_y_significances) - graph_border, 0)
    max_y = max(clean_y_significances) + graph_border
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    # Plots
    colors = plt.get_cmap("cool")

    ax.plot(
        plot_x,
        observed_significances_by_gaussian_fit,
        color=colors(0.75),
        linewidth=2,
        linestyle="--",
        label="gaussian fit significance",
    )
    ax.plot(
        plot_x,
        observed_significances,
        color=colors(0.5),
        label="observed significance",
        linewidth=2,
    )
    ax.fill_between(
        plot_x,
        np.clip(observed_significances_lower_confidence_bounds, a_min=0, a_max=max_y),
        np.clip(observed_significances_upper_confidence_bounds, a_min=0, a_max=max_y),
        color=colors(1),
        linewidth=2,
        alpha=0.1,
    )

    # Error bars
    ax.errorbar(
        plot_x,
        observed_significances,
        xerr=injected_significance_stds,
    )

    # Texting
    ax.set_xlabel(x_label, fontsize=21)
    ax.set_ylabel("measured significance", fontsize=21)
    ax.set_title("measured vs injected signal significance", fontsize=24)
    legend = ax.legend(loc="lower right", fontsize=20, fancybox=True, frameon=False)

    # Styling
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0.0)
    ax.tick_params(labelsize=20)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="lower"))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune="lower"))

    return fig


def plot_samples_over_background_sliced(
    context: ExecutionContext,
    background_solid_datasets: List[DataSet] = [],
    sample_hollow_datasets: List[DataSet] = [],
    observable: Optional[str] = None,
    title: str = "Sample over background",
    background_legends: List[str] = [],
    sample_legends: List[str] = [],
):
    """
    Generate two plots, both featuring historams of either sample over the background.
    Both are reconstructed to compensate for detector efficiency losses.
    """
    c = Carpenter(context)
    fig = c.figure()
    bins, _ = context.config.observable_bins(
        observable or context.config.detector__detect_observable_names[0]
    )

    datasets = sample_hollow_datasets + background_solid_datasets
    legends = sample_legends + background_legends
    ax = fig.add_subplot(111)
    for i, background in enumerate(datasets):
        utils__datset_histogram_sliced(
            ax=ax,
            bins=bins,
            dataset=background,
            along_observables=observable,
            label=legends[i],
            histtype="stepfilled" if i >= len(sample_hollow_datasets) else "step",
        )
    ax.set_title(title)

    return fig


def plot_data_generation_sliced(
    context: ExecutionContext,
    original_sample: DataSet,
    processed_sample: DataSet,
    observable: str,
):
    c = Carpenter(context)
    fig = c.figure()
    ax = fig.add_subplot(111)

    bins, _ = context.config.observable_bins(observable)

    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=original_sample,
        # the usual weights
        along_observables=observable,
        label="original sample",
        histtype="stepfilled",
        alpha=0.6,
    )
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=processed_sample,
        alternative_weights=np.ones(shape=(processed_sample.n_samples, 1)),
        along_observables=observable,
        label="detector affected sample",
        histtype="stepfilled",
        alpha=0.6,
    )
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=processed_sample,
        # the usual weights
        along_observables=observable,
        label="detector affected sample (weight adjusted)",
        histtype="step",
    )

    ax.set_title("Sample Generation Process Illustration", fontsize=24)
    ax.set_xlabel(f"{observable}", fontsize=20)
    ax.set_ylabel("number of events", fontsize=20)
    ax.legend()
    return fig


def _display_edges_by_observable(
    datasets: List[DataSet],
    observable_names: List[str],
    number_of_bins: int,
) -> dict[str, np.ndarray]:
    if number_of_bins <= 0:
        raise ValueError(
            f"Expected a positive number of display bins, got {number_of_bins}"
        )

    edges_by_observable = {}
    for observable_name in observable_names:
        values = np.concatenate(
            [
                utils__flatten_histogram_values(
                    dataset.slice_along_observable_names(observable_name)
                )
                for dataset in datasets
            ]
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(
                f"Cannot define display bins for {observable_name}: no finite values found."
            )

        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if minimum == maximum:
            padding = max(abs(minimum) * 0.05, 0.5)
            minimum -= padding
            maximum += padding
        edges_by_observable[observable_name] = np.linspace(
            minimum, maximum, number_of_bins + 1
        )

    return edges_by_observable


def _bins_for_observables(
    edges_by_observable: dict[str, np.ndarray],
    observable_names: List[str],
) -> Tuple[Union[np.ndarray, List[np.ndarray]], Union[np.ndarray, List[np.ndarray]]]:
    edges = [
        edges_by_observable[observable_name] for observable_name in observable_names
    ]
    centers = [
        0.5 * (observable_edges[:-1] + observable_edges[1:])
        for observable_edges in edges
    ]
    if len(observable_names) == 1:
        return edges[0], centers[0]
    return edges, centers


def _spanning_dataset_from_observable_values(
    values_by_observable: dict[str, np.ndarray],
    observable_names: List[str],
) -> DataSet:
    spanning_mesh = np.meshgrid(
        *[
            values_by_observable[observable_name]
            for observable_name in observable_names
        ],
        indexing="ij",
    )
    return DataSet(
        data=np.column_stack([dimension.ravel() for dimension in spanning_mesh]),
        observable_names=observable_names,
    )


def plot_prediction_process_1d(
    context: ExecutionContext,
    numerator_training: TrainLauncher.Training,
    denominator_training: TrainLauncher.Training,
    title: str = "Datasets Along the Process",
    along_observables: Union[List[str], str, None] = None,
) -> Figure:
    """
    Plot the SR/CR data distributions and the corresponding LFVDDP predictions.

    The numerator model's ``predict`` and ``predict_secondary`` outputs provide
    the combined e^f(1+eta) and e^g(1-eta) predictions. NPLM model compatibility
    is intentionally out of scope.
    """
    selected_observables = utils__prediction_process_observables(
        context, along_observables, required_dimensions=1
    )
    if not isinstance((config := context.config), TrainConfig):
        raise ValueError("The context config is not a TrainConfig.")
    if not isinstance(config, DatasetConfig):
        raise ValueError("The context config is not a DatasetConfig.")
    if not isinstance(config, DetectorConfig):
        raise ValueError("The context config is not a DetectorConfig.")
    if not isinstance(config, PlottingConfig):
        raise ValueError("The context config is not a PlottingConfig.")

    configured_observables = config.detector__detect_observable_names

    numerator_model = numerator_training.model
    denominator_model = denominator_training.model

    ndim = 1
    data_batch = numerator_training.data_batch
    datasets = data_batch.datasets
    a_sr = datasets[DataSet.DataSetCategory.A_SR]
    b_sr = datasets[DataSet.DataSetCategory.B_SR]
    a_cr = datasets[DataSet.DataSetCategory.A_CR]
    b_cr = datasets[DataSet.DataSetCategory.B_CR]
    sr_background = a_sr + b_sr
    cr_background = a_cr + b_cr

    c = Carpenter(context)
    fig = c.figure()
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.08,
        hspace=0.30,
        wspace=0.22,
    )
    fig.suptitle(title, fontsize=22)

    plot_colors = {
        "background": "gray",
        "f": "tab:blue",
        "g": "tab:orange",
        "eta_plus": "cornflowerblue",
        "eta_minus": "sandybrown",
    }
    prediction_linestyles = {
        "component": "-",
        "product": "-.",
        "denominator": "--",
    }

    display_edges_by_observable = _display_edges_by_observable(
        datasets=[data_batch.unified_data],
        observable_names=configured_observables,
        number_of_bins=config.plot__prediction_process_number_of_bins,
    )
    bins, bin_centers = _bins_for_observables(
        display_edges_by_observable, selected_observables
    )

    sr_distribution_ax = utils__add_subplot_sliced(fig, (2, 2, 1), ndim)
    cr_distribution_ax = utils__add_subplot_sliced(fig, (2, 2, 2), ndim)
    sr_prediction_ax = utils__add_subplot_sliced(fig, (2, 2, 3), ndim)
    cr_prediction_ax = utils__add_subplot_sliced(fig, (2, 2, 4), ndim)

    normalize_top_distributions = (
        config.plot__prediction_process_normalize_each_prediction
    )

    utils__plot_region_histograms_sliced(
        ax=sr_distribution_ax,
        sample_a=a_sr,
        sample_b=b_sr,
        background=sr_background,
        bins=bins,
        along_observables=selected_observables,
        region_name="SR",
        background_color=plot_colors["background"],
        sample_a_color=plot_colors["f"],
        sample_b_color=plot_colors["g"],
        normalize_distributions=normalize_top_distributions,
    )
    utils__plot_region_histograms_sliced(
        ax=cr_distribution_ax,
        sample_a=a_cr,
        sample_b=b_cr,
        background=cr_background,
        bins=bins,
        along_observables=selected_observables,
        region_name="CR",
        background_color=plot_colors["background"],
        sample_a_color=plot_colors["f"],
        sample_b_color=plot_colors["g"],
        normalize_distributions=normalize_top_distributions,
    )

    def weighted_distribution_predictions(
        reference_background: DataSet,
        prediction_specs: Tuple[
            Tuple[np.ndarray, float, str, str, str], ...
        ],
    ) -> List[Tuple[np.ndarray, str, str, str]]:
        return [
            (
                prediction_to_sample_ndf_hypothesis_weights(
                    model_prediction=prediction,
                    predicted_distribution_size=ndf_n_samples,
                    reference_ndf_estimation=reference_background,
                ),
                label,
                color,
                marker,
            )
            for (
                prediction,
                ndf_n_samples,
                label,
                color,
                marker,
            ) in prediction_specs
        ]

    sr_exp_f_eta_plus = utils__model_prediction_values(
        numerator_model.predict, sr_background
    )
    sr_exp_g_eta_minus = utils__model_prediction_values(
        numerator_model.predict_secondary, sr_background
    )
    sr_product_predictions = (
        (
            sr_exp_f_eta_plus,
            a_sr.n_samples,
            r"$e^{f(x)}(1+\eta(x))$ prediction",
            plot_colors["f"],
            "o",
        ),
        (
            sr_exp_g_eta_minus,
            b_sr.n_samples,
            r"$e^{g(x)}(1-\eta(x))$ prediction",
            plot_colors["g"],
            "s",
        ),
    )
    utils__plot_weighted_histogram_predictions_sliced(
        ax=sr_distribution_ax,
        reference_dataset=sr_background,
        predictions=weighted_distribution_predictions(
            sr_background, sr_product_predictions
        ),
        bins=bins,
        bin_centers=bin_centers,
        along_observables=selected_observables,
        normalize_each_prediction=normalize_top_distributions,
    )

    cr_eta = utils__model_prediction_values(
        denominator_model.predict_eta, cr_background
    )
    cr_eta_predictions = (
        (
            1.0 + cr_eta,
            a_cr.n_samples,
            r"null hypothesis $1+\eta(x)$ prediction",
            plot_colors["eta_plus"],
            "o",
        ),
        (
            1.0 - cr_eta,
            b_cr.n_samples,
            r"null hypothesis $1-\eta(x)$ prediction",
            plot_colors["eta_minus"],
            "s",
        ),
    )
    utils__plot_weighted_histogram_predictions_sliced(
        ax=cr_distribution_ax,
        reference_dataset=cr_background,
        predictions=weighted_distribution_predictions(
            cr_background, cr_eta_predictions
        ),
        bins=bins,
        bin_centers=bin_centers,
        along_observables=selected_observables,
        normalize_each_prediction=normalize_top_distributions,
    )

    distribution_axes = (sr_distribution_ax, cr_distribution_ax)
    utils__synchronize_output_axis_limits(list(distribution_axes), ndim)

    for panel in distribution_axes:
        panel.legend(fontsize=8)

    detector_effect = denominator_training.detector_effect
    detector_bins_by_observable = {
        observable_name: detector_effect.get_observable_bins(observable_name)
        for observable_name in configured_observables
    }
    detector_edges_by_observable = {
        observable_name: detector_bins[0]
        for observable_name, detector_bins in detector_bins_by_observable.items()
    }
    detector_centers_by_observable = {
        observable_name: detector_bins[1]
        for observable_name, detector_bins in detector_bins_by_observable.items()
    }

    def dense_display_axis_values(
        display_edges: np.ndarray,
        detector_edges: np.ndarray,
    ) -> np.ndarray:
        step = np.min(np.diff(display_edges)) / 10.0
        return np.unique(
            np.concatenate(
                (
                    np.arange(display_edges[0], display_edges[-1], step),
                    display_edges,
                    detector_edges[
                        (detector_edges >= display_edges[0])
                        & (detector_edges <= display_edges[-1])
                    ],
                )
            )
        )

    prediction_values_by_observable = {
        observable_name: (
            dense_display_axis_values(
                display_edges_by_observable[observable_name],
                detector_edges_by_observable[observable_name],
            )
            if observable_name in selected_observables
            else detector_centers_by_observable[observable_name]
        )
        for observable_name in configured_observables
    }
    prediction_spanning_dataset = _spanning_dataset_from_observable_values(
        values_by_observable=prediction_values_by_observable,
        observable_names=configured_observables,
    )

    spanning_exp_f_eta_plus = utils__model_prediction_values(
        numerator_model.predict, prediction_spanning_dataset
    )
    spanning_exp_g_eta_minus = utils__model_prediction_values(
        numerator_model.predict_secondary, prediction_spanning_dataset
    )
    nuisance_numerator_eta = utils__model_prediction_values(
        numerator_model.predict_eta, prediction_spanning_dataset
    )
    nuisance_denominator_eta = utils__model_prediction_values(
        denominator_model.predict_eta, prediction_spanning_dataset
    )
    spanning_exp_f = utils__remove_eta_from_prediction_values(
        prediction_values=spanning_exp_f_eta_plus,
        eta_values=nuisance_numerator_eta,
        eta_sign=1.0,
    )
    spanning_exp_g = utils__remove_eta_from_prediction_values(
        prediction_values=spanning_exp_g_eta_minus,
        eta_values=nuisance_numerator_eta,
        eta_sign=-1.0,
    )

    sr_prediction_specs = {
        "exp_f": (
            r"signal hypothesis $e^{f(x)}$",
            spanning_exp_f,
            plot_colors["f"],
            prediction_linestyles["product"],
        ),
        "exp_g": (
            r"signal hypothesis $e^{g(x)}$",
            spanning_exp_g,
            plot_colors["g"],
            prediction_linestyles["product"],
        ),
        "exp_f_eta_plus": (
            r"signal hypothesis $e^{f(x)}(1+\eta(x))$",
            spanning_exp_f_eta_plus,
            plot_colors["f"],
            prediction_linestyles["component"],
        ),
        "exp_g_eta_minus": (
            r"signal hypothesis $e^{g(x)}(1-\eta(x))$",
            spanning_exp_g_eta_minus,
            plot_colors["g"],
            prediction_linestyles["component"],
        ),
    }

    def nuisance_prediction_specs(
        numerator_eta: np.ndarray,
        denominator_eta: np.ndarray,
    ) -> dict[str, Tuple[str, np.ndarray, str, str]]:
        return {
            "numerator_eta_plus": (
                r"signal hypothesis $1+\eta(x)$",
                1.0 + numerator_eta,
                plot_colors["eta_plus"],
                prediction_linestyles["component"],
            ),
            "numerator_eta_minus": (
                r"signal hypothesis $1-\eta(x)$",
                1.0 - numerator_eta,
                plot_colors["eta_minus"],
                prediction_linestyles["component"],
            ),
            "denominator_eta_plus": (
                r"null hypothesis $1+\eta(x)$",
                1.0 + denominator_eta,
                plot_colors["eta_plus"],
                prediction_linestyles["denominator"],
            ),
            "denominator_eta_minus": (
                r"null hypothesis $1-\eta(x)$",
                1.0 - denominator_eta,
                plot_colors["eta_minus"],
                prediction_linestyles["denominator"],
            ),
        }

    cr_prediction_specs = nuisance_prediction_specs(
        nuisance_numerator_eta,
        nuisance_denominator_eta,
    )

    def project_predictions(
        prediction_specs: dict[str, Tuple[str, np.ndarray, str, str]],
        spanning_dataset: DataSet,
    ) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {
            key: utils__project_prediction_values_sliced(
                values=specification[1],
                spanning_dataset=spanning_dataset,
                along_observables=selected_observables,
            )
            for key, specification in prediction_specs.items()
        }

    projected_sr_predictions = project_predictions(
        sr_prediction_specs, prediction_spanning_dataset
    )
    projected_cr_predictions = project_predictions(
        cr_prediction_specs, prediction_spanning_dataset
    )

    def prediction_axis_limits(
        projected_predictions: dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[float, float]:
        finite_prediction_chunks = []
        for _, contour in projected_predictions.values():
            flattened_contour = utils__flatten_histogram_values(contour)
            finite_prediction_chunks.append(
                flattened_contour[np.isfinite(flattened_contour)]
            )
        finite_prediction_values = np.concatenate(finite_prediction_chunks)
        prediction_minimum = min(1.0, float(np.min(finite_prediction_values)))
        prediction_maximum = max(1.0, float(np.max(finite_prediction_values)))
        prediction_span = prediction_maximum - prediction_minimum
        prediction_padding = (
            0.05 * prediction_span if prediction_span > 0 else 0.05
        )
        return (
            prediction_minimum - prediction_padding,
            prediction_maximum + prediction_padding,
        )

    sr_prediction_limits = prediction_axis_limits(projected_sr_predictions)
    cr_prediction_limits = prediction_axis_limits(projected_cr_predictions)

    def projected_specs(
        prediction_specs: dict[str, Tuple[str, np.ndarray, str, str]],
        projected_predictions: dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> List[Tuple[np.ndarray, np.ndarray, str, str, str]]:
        return [
            (
                *projected_predictions[prediction_key],
                specification[0],
                specification[2],
                specification[3],
            )
            for prediction_key, specification in prediction_specs.items()
        ]

    utils__plot_model_predictions_sliced(
        ax=sr_prediction_ax,
        predictions=projected_specs(
            sr_prediction_specs, projected_sr_predictions
        ),
        bins=bins,
        along_observables=selected_observables,
        prediction_limits=sr_prediction_limits,
        title="SR predictions",
    )
    utils__plot_model_predictions_sliced(
        ax=cr_prediction_ax,
        predictions=projected_specs(
            cr_prediction_specs, projected_cr_predictions
        ),
        bins=bins,
        along_observables=selected_observables,
        prediction_limits=cr_prediction_limits,
        title="CR predictions",
    )

    return fig


def plot_prediction_process_2d(
    context: ExecutionContext,
    numerator_training: TrainLauncher.Training,
    denominator_training: TrainLauncher.Training,
    title: str = "Datasets Along the Process",
    along_observables: Union[List[str], str, None] = None,
) -> Figure:
    """
    Plot the SR/CR data distributions and the corresponding LFVDDP predictions.

    The numerator model's ``predict`` and ``predict_secondary`` outputs provide
    the combined e^f(1+eta) and e^g(1-eta) predictions. NPLM model compatibility
    is intentionally out of scope.
    """
    selected_observables = utils__prediction_process_observables(
        context, along_observables, required_dimensions=2
    )
    if not isinstance((config := context.config), TrainConfig):
        raise ValueError("The context config is not a TrainConfig.")
    if not isinstance(config, DatasetConfig):
        raise ValueError("The context config is not a DatasetConfig.")
    if not isinstance(config, DetectorConfig):
        raise ValueError("The context config is not a DetectorConfig.")
    if not isinstance(config, PlottingConfig):
        raise ValueError("The context config is not a PlottingConfig.")

    configured_observables = config.detector__detect_observable_names

    numerator_model = numerator_training.model
    denominator_model = denominator_training.model

    ndim = 2
    data_batch = numerator_training.data_batch
    datasets = data_batch.datasets
    a_sr = datasets[DataSet.DataSetCategory.A_SR]
    b_sr = datasets[DataSet.DataSetCategory.B_SR]
    a_cr = datasets[DataSet.DataSetCategory.A_CR]
    b_cr = datasets[DataSet.DataSetCategory.B_CR]
    sr_background = a_sr + b_sr
    cr_background = a_cr + b_cr

    c = Carpenter(context)
    fig = c.figure()
    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.90,
        bottom=0.08,
        hspace=0.30,
        wspace=0.22,
    )
    fig.suptitle(title, fontsize=22)

    plot_colors = {
        "background": "gray",
        "f": "tab:blue",
        "g": "tab:orange",
        "eta_plus": "cornflowerblue",
        "eta_minus": "sandybrown",
    }
    prediction_linestyles = {
        "component": "-",
        "product": "-.",
        "denominator": "--",
    }

    display_edges_by_observable = _display_edges_by_observable(
        datasets=[data_batch.unified_data],
        observable_names=configured_observables,
        number_of_bins=config.plot__prediction_process_number_of_bins,
    )
    bins, bin_centers = _bins_for_observables(
        display_edges_by_observable, selected_observables
    )

    sr_distribution_ax = utils__add_subplot_sliced(fig, (2, 2, 1), ndim)
    cr_distribution_ax = utils__add_subplot_sliced(fig, (2, 2, 2), ndim)
    sr_prediction_ax = utils__add_subplot_sliced(fig, (2, 2, 3), ndim)
    cr_prediction_ax = utils__add_subplot_sliced(fig, (2, 2, 4), ndim)

    normalize_top_distributions = (
        config.plot__prediction_process_normalize_each_prediction
    )

    utils__plot_region_histogram_meshes_2d(
        ax=sr_distribution_ax,
        sample_a=a_sr,
        sample_b=b_sr,
        background=sr_background,
        bins=bins,
        along_observables=selected_observables,
        region_name="SR",
        background_color=plot_colors["background"],
        sample_a_color=plot_colors["f"],
        sample_b_color=plot_colors["g"],
        normalize_distributions=normalize_top_distributions,
    )
    utils__plot_region_histogram_meshes_2d(
        ax=cr_distribution_ax,
        sample_a=a_cr,
        sample_b=b_cr,
        background=cr_background,
        bins=bins,
        along_observables=selected_observables,
        region_name="CR",
        background_color=plot_colors["background"],
        sample_a_color=plot_colors["f"],
        sample_b_color=plot_colors["g"],
        normalize_distributions=normalize_top_distributions,
    )

    def weighted_distribution_predictions(
        reference_background: DataSet,
        prediction_specs: Tuple[
            Tuple[np.ndarray, float, str, str, str], ...
        ],
    ) -> List[Tuple[np.ndarray, str, str, str]]:
        return [
            (
                prediction_to_sample_ndf_hypothesis_weights(
                    model_prediction=prediction,
                    predicted_distribution_corrected_size=ndf_corrected_n_samples,
                    reference_ndf_estimation=reference_background,
                ),
                label,
                color,
                marker,
            )
            for (
                prediction,
                ndf_corrected_n_samples,
                label,
                color,
                marker,
            ) in prediction_specs
        ]

    sr_exp_f_eta_plus = utils__model_prediction_values(
        numerator_model.predict, sr_background
    )
    sr_exp_g_eta_minus = utils__model_prediction_values(
        numerator_model.predict_secondary, sr_background
    )
    sr_product_predictions = (
        (
            sr_exp_f_eta_plus,
            a_sr.corrected_n_samples,
            r"$e^{f(x)}(1+\eta(x))$ prediction",
            plot_colors["f"],
            "o",
        ),
        (
            sr_exp_g_eta_minus,
            b_sr.corrected_n_samples,
            r"$e^{g(x)}(1-\eta(x))$ prediction",
            plot_colors["g"],
            "s",
        ),
    )
    utils__plot_weighted_histogram_predictions_sliced(
        ax=sr_distribution_ax,
        reference_dataset=sr_background,
        predictions=weighted_distribution_predictions(
            sr_background, sr_product_predictions
        ),
        bins=bins,
        bin_centers=bin_centers,
        along_observables=selected_observables,
        normalize_each_prediction=normalize_top_distributions,
    )

    cr_eta = utils__model_prediction_values(
        denominator_model.predict_eta, cr_background
    )
    cr_eta_predictions = (
        (
            1.0 + cr_eta,
            a_cr.corrected_n_samples,
            r"null hypothesis $1+\eta(x)$ prediction",
            plot_colors["eta_plus"],
            "o",
        ),
        (
            1.0 - cr_eta,
            b_cr.corrected_n_samples,
            r"null hypothesis $1-\eta(x)$ prediction",
            plot_colors["eta_minus"],
            "s",
        ),
    )
    utils__plot_weighted_histogram_predictions_sliced(
        ax=cr_distribution_ax,
        reference_dataset=cr_background,
        predictions=weighted_distribution_predictions(
            cr_background, cr_eta_predictions
        ),
        bins=bins,
        bin_centers=bin_centers,
        along_observables=selected_observables,
        normalize_each_prediction=normalize_top_distributions,
    )

    distribution_axes = (sr_distribution_ax, cr_distribution_ax)
    utils__synchronize_output_axis_limits(list(distribution_axes), ndim)

    for panel in distribution_axes:
        panel.legend(fontsize=8)

    detector_effect = denominator_training.detector_effect
    detector_bins_by_observable = {
        observable_name: detector_effect.get_observable_bins(observable_name)
        for observable_name in configured_observables
    }
    detector_edges_by_observable = {
        observable_name: detector_bins[0]
        for observable_name, detector_bins in detector_bins_by_observable.items()
    }
    detector_centers_by_observable = {
        observable_name: detector_bins[1]
        for observable_name, detector_bins in detector_bins_by_observable.items()
    }

    def dense_display_axis_values(
        display_edges: np.ndarray,
        detector_edges: np.ndarray,
    ) -> np.ndarray:
        step = np.min(np.diff(display_edges)) / 10.0
        return np.unique(
            np.concatenate(
                (
                    np.arange(display_edges[0], display_edges[-1], step),
                    display_edges,
                    detector_edges[
                        (detector_edges >= display_edges[0])
                        & (detector_edges <= display_edges[-1])
                    ],
                )
            )
        )

    prediction_values_by_observable = {
        observable_name: (
            dense_display_axis_values(
                display_edges_by_observable[observable_name],
                detector_edges_by_observable[observable_name],
            )
            if observable_name in selected_observables
            else detector_centers_by_observable[observable_name]
        )
        for observable_name in configured_observables
    }
    prediction_spanning_dataset = _spanning_dataset_from_observable_values(
        values_by_observable=prediction_values_by_observable,
        observable_names=configured_observables,
    )

    spanning_exp_f_eta_plus = utils__model_prediction_values(
        numerator_model.predict, prediction_spanning_dataset
    )
    spanning_exp_g_eta_minus = utils__model_prediction_values(
        numerator_model.predict_secondary, prediction_spanning_dataset
    )
    nuisance_numerator_eta = utils__model_prediction_values(
        numerator_model.predict_eta, prediction_spanning_dataset
    )
    nuisance_denominator_eta = utils__model_prediction_values(
        denominator_model.predict_eta, prediction_spanning_dataset
    )
    spanning_exp_f = utils__remove_eta_from_prediction_values(
        prediction_values=spanning_exp_f_eta_plus,
        eta_values=nuisance_numerator_eta,
        eta_sign=1.0,
    )
    spanning_exp_g = utils__remove_eta_from_prediction_values(
        prediction_values=spanning_exp_g_eta_minus,
        eta_values=nuisance_numerator_eta,
        eta_sign=-1.0,
    )

    sr_prediction_specs = {
        "exp_f": (
            r"signal hypothesis $e^{f(x)}$",
            spanning_exp_f,
            plot_colors["f"],
            prediction_linestyles["product"],
        ),
        "exp_g": (
            r"signal hypothesis $e^{g(x)}$",
            spanning_exp_g,
            plot_colors["g"],
            prediction_linestyles["product"],
        ),
        "exp_f_eta_plus": (
            r"signal hypothesis $e^{f(x)}(1+\eta(x))$",
            spanning_exp_f_eta_plus,
            plot_colors["f"],
            prediction_linestyles["component"],
        ),
        "exp_g_eta_minus": (
            r"signal hypothesis $e^{g(x)}(1-\eta(x))$",
            spanning_exp_g_eta_minus,
            plot_colors["g"],
            prediction_linestyles["component"],
        ),
    }

    def nuisance_prediction_specs(
        numerator_eta: np.ndarray,
        denominator_eta: np.ndarray,
    ) -> dict[str, Tuple[str, np.ndarray, str, str]]:
        return {
            "numerator_eta_plus": (
                r"signal hypothesis $1+\eta(x)$",
                1.0 + numerator_eta,
                plot_colors["eta_plus"],
                prediction_linestyles["component"],
            ),
            "numerator_eta_minus": (
                r"signal hypothesis $1-\eta(x)$",
                1.0 - numerator_eta,
                plot_colors["eta_minus"],
                prediction_linestyles["component"],
            ),
            "denominator_eta_plus": (
                r"null hypothesis $1+\eta(x)$",
                1.0 + denominator_eta,
                plot_colors["eta_plus"],
                prediction_linestyles["denominator"],
            ),
            "denominator_eta_minus": (
                r"null hypothesis $1-\eta(x)$",
                1.0 - denominator_eta,
                plot_colors["eta_minus"],
                prediction_linestyles["denominator"],
            ),
        }

    cr_prediction_specs = nuisance_prediction_specs(
        nuisance_numerator_eta,
        nuisance_denominator_eta,
    )

    def project_predictions(
        prediction_specs: dict[str, Tuple[str, np.ndarray, str, str]],
        spanning_dataset: DataSet,
    ) -> dict[str, Tuple[np.ndarray, np.ndarray]]:
        return {
            key: utils__project_prediction_values_sliced(
                values=specification[1],
                spanning_dataset=spanning_dataset,
                along_observables=selected_observables,
            )
            for key, specification in prediction_specs.items()
        }

    projected_sr_predictions = project_predictions(
        sr_prediction_specs, prediction_spanning_dataset
    )
    projected_cr_predictions = project_predictions(
        cr_prediction_specs, prediction_spanning_dataset
    )

    def prediction_axis_limits(
        projected_predictions: dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[float, float]:
        finite_prediction_chunks = []
        for _, contour in projected_predictions.values():
            flattened_contour = utils__flatten_histogram_values(contour)
            finite_prediction_chunks.append(
                flattened_contour[np.isfinite(flattened_contour)]
            )
        finite_prediction_values = np.concatenate(finite_prediction_chunks)
        prediction_minimum = min(1.0, float(np.min(finite_prediction_values)))
        prediction_maximum = max(1.0, float(np.max(finite_prediction_values)))
        prediction_span = prediction_maximum - prediction_minimum
        prediction_padding = (
            0.05 * prediction_span if prediction_span > 0 else 0.05
        )
        return (
            prediction_minimum - prediction_padding,
            prediction_maximum + prediction_padding,
        )

    sr_prediction_limits = prediction_axis_limits(projected_sr_predictions)
    cr_prediction_limits = prediction_axis_limits(projected_cr_predictions)

    def projected_specs(
        prediction_specs: dict[str, Tuple[str, np.ndarray, str, str]],
        projected_predictions: dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> List[Tuple[np.ndarray, np.ndarray, str, str, str]]:
        return [
            (
                *projected_predictions[prediction_key],
                specification[0],
                specification[2],
                specification[3],
            )
            for prediction_key, specification in prediction_specs.items()
        ]

    utils__plot_model_predictions_sliced(
        ax=sr_prediction_ax,
        predictions=projected_specs(
            sr_prediction_specs, projected_sr_predictions
        ),
        bins=bins,
        along_observables=selected_observables,
        prediction_limits=sr_prediction_limits,
        title="SR predictions",
    )
    utils__plot_model_predictions_sliced(
        ax=cr_prediction_ax,
        predictions=projected_specs(
            cr_prediction_specs, projected_cr_predictions
        ),
        bins=bins,
        along_observables=selected_observables,
        prediction_limits=cr_prediction_limits,
        title="CR predictions",
    )

    return fig
