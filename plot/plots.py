from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

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
from data_tools.detector.detector_effect import DetectorEffect
from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_continuous,
    calc_median_t_significance_relative_to_background,
    calc_t_significance_by_gaussian_fit_percentile,
    calc_t_significance_relative_to_background,
)
from frame.aggregate import ResultAggregator, utils__get_signal_dataset_parameters
from frame.context.execution_context import ExecutionContext
from frame.file_structure import CONTEXT_FILE_NAME
from neural_networks.utils import (
    ContextedModel,
    prediction_to_sample_ndf_hypothesis_weights,
)
from plot.carpenter import Carpenter
from plot.plot_utils import (
    HandlerCircle,
    HandlerRect,
    utils__contour_model_prediction,
    utils__datset_histogram_sliced,
    utils__flatten_histogram_values,
    utils__samples_over_background_histograms_sliced,
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
    The funcion creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    The percentile lines for the target chi2 distribution are shown as a reference.

    patience:      (int) interval between two check points (epochs).
    tvalues_check: (numpy array shape (N_toys, N_check_points)) array of t=-2*loss
    df:            (int) chi2 degrees of freedom
    """
    if not isinstance(config := context.config, PlottingConfig):
        raise ValueError(
            f"Expected context.config to be of type {PlottingConfig}, got {type(config)}"
        )

    # Training results aggregation
    agg = ResultAggregator(Path(config.plot__target_run_parent_directory))
    all_model_t_test_statistics = agg.all_test_statistics
    epochs = agg.all_epochs

    # Framing
    c = Carpenter(context)
    fig = c.figure()
    ax = fig.add_subplot(111)

    # Drawing
    legend = []
    quantiles = [2.5, 25, 50, 75, 97.5]
    percentiles = np.apply_along_axis(
        lambda x: np.nanpercentile(x, quantiles), 0, all_model_t_test_statistics
    )
    colors = ["violet", "hotpink", "mediumvioletred", "mediumorchid", "darkviolet"]

    # Training percentile progression
    for j in range(percentiles.shape[0]):
        plt.plot(epochs, percentiles[j, :], linewidth=3, color=colors[j])
        legend.append(str(quantiles[j]) + "% quantile")

    # chi2 reference
    for j in range(percentiles.shape[0]):
        plt.plot(
            epochs,
            chi2.ppf(
                quantiles[j] / 100.0,
                df=model_degrees_of_freedom(config),
                loc=0,
                scale=1,
            )
            * np.ones_like(epochs),
            color=colors[j],
            ls="--",
            linewidth=1,
        )
        if j == 0:
            legend.append(
                "Target " + r"$\chi^2($" + str(model_degrees_of_freedom(config)) + ")"
            )

    # Labeling
    plt.title(r"$\chi^2$ percentile progression", fontsize=24)

    if np.any(np.isnan(all_model_t_test_statistics)):
        legend.append(
            f"Nan percent: {np.count_nonzero(np.isnan(all_model_t_test_statistics)) / all_model_t_test_statistics.size * 100:.2f}"
        )
    plt.legend(legend, frameon=False, markerscale=0)

    plt.xlabel("Training Epochs", fontsize=22)
    plt.ylabel("t", fontsize=22)
    plt.xlim(0, np.max(epochs))
    plt.ylim(0, np.nanmax(percentiles))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(18)

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
    critical_mass_t = t > fifth_percentile
    distribution_std = np.std(t[critical_mass_t])
    distribution_mean = np.mean(t[critical_mass_t])
    n_std = 6
    did_not_converge = t < (distribution_mean - n_std * distribution_std)
    if cut_non_converged:
        t = t[~did_not_converge]

    # Limits
    chi2_begin = 0
    chi2_end = chi2.ppf(0.9999, chi2_dof := model_degrees_of_freedom(config))
    xmin = min(t)
    xmax = max(t)

    # plot distribution histogram
    histogram_bins = np.linspace(0, xmax, number_of_bins + 1)
    histogram_bin_width = (xmax - xmin) * 1.0 / number_of_bins
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
        weights=np.ones_like(t) * (number_of_bins / ((xmax - xmin) * t.shape[0])),
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
        label=f"$\chi^{2}_{{{chi2_dof}}}$",
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
        (label, f"$\chi^{2}_{{{chi2_dof}}}$"),
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
    ax.set_ylim(0, top=max(h + y_error))
    ax.set_xlim(0, xmax)
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


def _model_prediction_specs(
    trained_model: ContextedModel,
    base_legend: str,
    primary_color: str,
    secondary_color: str,
) -> List[Tuple[Callable[[DataSet], np.ndarray], str, str]]:
    predictions = [(trained_model.predict, base_legend, primary_color)]
    if hasattr(trained_model, "predict_secondary"):
        predictions.append(
            (
                trained_model.predict_secondary,
                f"{base_legend} secondary",
                secondary_color,
            )
        )
    return predictions


def _model_contour_specs(
    trained_model: ContextedModel,
    base_legend: str,
    primary_color: str,
    secondary_color: str,
    eta_color: str,
) -> List[
    Tuple[Callable[[DataSet], np.ndarray], str, str, Callable[[np.ndarray], np.ndarray]]
]:
    predict_f = getattr(trained_model, "predict_f", None)
    predict_g = getattr(trained_model, "predict_g", None)
    predict_eta = getattr(trained_model, "predict_eta", None)
    if callable(predict_f) and callable(predict_g):
        specs = [
            (predict_f, f"{base_legend} (f)", primary_color, np.exp),
            (predict_g, f"{base_legend} (g)", secondary_color, np.exp),
        ]
        if callable(predict_eta):
            specs.append((predict_eta, f"{base_legend} (eta)", eta_color, np.asarray))
        return specs

    predict = getattr(trained_model, "predict", None)
    if callable(predict):
        return [(predict, base_legend, primary_color, np.exp)]

    raise AttributeError(
        f"{trained_model.__class__.__name__} must expose predict_f/predict_g or predict to be plotted."
    )


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


def _spanning_dataset_from_centers(
    centers_by_observable: dict[str, np.ndarray],
    observable_names: List[str],
) -> DataSet:
    spanning_mesh = np.meshgrid(
        *[
            centers_by_observable[observable_name]
            for observable_name in observable_names
        ],
        indexing="ij",
    )
    return DataSet(
        data=np.column_stack([dimension.ravel() for dimension in spanning_mesh]),
        observable_names=observable_names,
    )


def plot_prediction_process_sliced(
    context: ExecutionContext,
    numerator_training: TrainLauncher.Training,
    denominator_training: TrainLauncher.Training,
    title: str = "Datasets Along the Process",
    along_observables: Union[List[str], str, None] = None,
) -> Figure:
    """
    Plot the SR/CR data distributions and the corresponding LFVDDP predictions.

    The numerator model's ``predict`` and ``predict_secondary`` outputs are treated
    directly as f and g. NPLM model compatibility is intentionally out of scope.
    """
    if not isinstance((config := context.config), TrainConfig):
        raise ValueError("The context config is not a TrainConfig.")
    if not isinstance(config, DatasetConfig):
        raise ValueError("The context config is not a DatasetConfig.")
    if not isinstance(config, DetectorConfig):
        raise ValueError("The context config is not a DetectorConfig.")
    if not isinstance(config, PlottingConfig):
        raise ValueError("The context config is not a PlottingConfig.")

    configured_observables = config.detector__detect_observable_names
    if along_observables is None:
        selected_observables = configured_observables[:2]
    elif isinstance(along_observables, str):
        selected_observables = [along_observables]
    else:
        selected_observables = list(along_observables)
    if len(selected_observables) > 2:
        raise ValueError("Cannot plot more than 2 observables in a single plot.")

    numerator_model = numerator_training.model
    denominator_model = denominator_training.model

    ndim = len(selected_observables)
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
        "eta": "darkviolet",
        "eta_plus": "mediumpurple",
        "eta_minus": "plum",
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
    display_centers_by_observable = {
        observable_name: 0.5 * (observable_edges[:-1] + observable_edges[1:])
        for observable_name, observable_edges in display_edges_by_observable.items()
    }
    bins, bin_centers = _bins_for_observables(
        display_edges_by_observable, selected_observables
    )
    contour_spanning_dataset = _spanning_dataset_from_centers(
        centers_by_observable=display_centers_by_observable,
        observable_names=configured_observables,
    )

    def add_panel(position: int) -> plt.Axes:
        if ndim == 1:
            return fig.add_subplot(2, 2, position)
        panel = fig.add_subplot(2, 2, position, projection="3d")
        panel.view_init(elev=28, azim=-58)
        panel.set_box_aspect((1.2, 1.2, 0.9))
        return panel

    sr_distribution_ax = add_panel(1)
    cr_distribution_ax = add_panel(2)
    sr_prediction_ax = add_panel(3)
    cr_prediction_ax = add_panel(4)

    def set_observable_axes(panel: plt.Axes, output_label: str) -> None:
        if ndim == 1:
            panel.set_xlim(bins[0], bins[-1])
            panel.set_xlabel(selected_observables[0])
            panel.set_ylabel(output_label)
            return
        panel.set_xlim(bins[0][0], bins[0][-1])
        panel.set_ylim(bins[1][0], bins[1][-1])
        panel.set_xlabel(selected_observables[0])
        panel.set_ylabel(selected_observables[1])
        panel.set_zlabel(output_label)

    def draw_region_distribution(
        panel: plt.Axes,
        sample_a: DataSet,
        sample_b: DataSet,
        background: DataSet,
        region_name: str,
    ) -> None:
        distribution_specs = (
            (
                background,
                f"A-{region_name} + B-{region_name} (background)",
                plot_colors["background"],
                "stepfilled",
                0.28,
                1.0,
            ),
            (sample_a, f"A-{region_name}", plot_colors["f"], "step", 1.0, 1.8),
            (sample_b, f"B-{region_name}", plot_colors["g"], "step", 1.0, 1.8),
        )
        for dataset, label, color, histtype, alpha, linewidth in distribution_specs:
            utils__datset_histogram_sliced(
                ax=panel,
                bins=bins,
                dataset=dataset,
                along_observables=selected_observables,
                label=label,
                color=color,
                histtype=histtype,
                alpha=alpha,
                lw=linewidth,
            )
        if ndim == 2:
            panel.set_zlim(0.1, max(1.0, panel.get_zlim()[1]))
            panel.set_zscale("log")
            panel.zaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
            panel.zaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
        set_observable_axes(panel, "number of events")
        panel.set_title(f"{region_name} distributions")

    draw_region_distribution(sr_distribution_ax, a_sr, b_sr, sr_background, "SR")
    draw_region_distribution(cr_distribution_ax, a_cr, b_cr, cr_background, "CR")

    def prediction_values(
        model: ContextedModel,
        method_name: str,
        dataset: DataSet,
    ) -> np.ndarray:
        return utils__flatten_histogram_values(getattr(model, method_name)(dataset))

    sr_f = prediction_values(numerator_model, "predict", sr_background)
    sr_g = prediction_values(numerator_model, "predict_secondary", sr_background)
    sr_eta = prediction_values(numerator_model, "predict_eta", sr_background)
    sr_product_predictions = (
        (
            sr_f * (1.0 + sr_eta),
            r"$f(x)(1+\eta(x))$ prediction",
            plot_colors["f"],
            "o",
        ),
        (
            sr_g * (1.0 - sr_eta),
            r"$g(x)(1-\eta(x))$ prediction",
            plot_colors["g"],
            "s",
        ),
    )
    sr_background_data = np.asarray(
        sr_background.slice_along_observable_names(selected_observables)
    ).reshape(sr_background.n_samples, ndim)

    if ndim == 2:
        prediction_xx, prediction_yy = np.meshgrid(
            bin_centers[0], bin_centers[1], indexing="ij"
        )
    for prediction, label, color, marker in sr_product_predictions:
        hypothesis_weights = prediction_to_sample_ndf_hypothesis_weights(
            model_prediction=prediction,
            predicted_distribution_corrected_size=sr_background.corrected_n_samples,
            reference_ndf_estimation=sr_background,
        )
        flattened_weights = utils__flatten_histogram_values(hypothesis_weights)
        if ndim == 1:
            predicted_counts, _ = np.histogram(
                sr_background_data[:, 0],
                bins=bins,
                weights=flattened_weights,
            )
            sr_distribution_ax.scatter(
                bin_centers,
                predicted_counts,
                label=label,
                color=color,
                marker=marker,
                s=28,
                edgecolor="black",
                linewidth=0.5,
            )
        else:
            predicted_counts, _, _ = np.histogram2d(
                sr_background_data[:, 0],
                sr_background_data[:, 1],
                bins=bins,
                weights=flattened_weights,
            )
            positive_counts = predicted_counts.ravel() > 0
            sr_distribution_ax.scatter(
                prediction_xx.ravel()[positive_counts],
                prediction_yy.ravel()[positive_counts],
                predicted_counts.ravel()[positive_counts],
                label=label,
                color=color,
                marker=marker,
                s=22,
                edgecolor="black",
                linewidth=0.4,
            )

    distribution_axes = (sr_distribution_ax, cr_distribution_ax)
    if ndim == 1:
        shared_distribution_limits = (
            min(panel.get_ylim()[0] for panel in distribution_axes),
            max(panel.get_ylim()[1] for panel in distribution_axes),
        )
        for panel in distribution_axes:
            panel.set_ylim(shared_distribution_limits)
    else:
        shared_distribution_limits = (
            min(panel.get_zlim()[0] for panel in distribution_axes),
            max(panel.get_zlim()[1] for panel in distribution_axes),
        )
        for panel in distribution_axes:
            panel.set_zlim(shared_distribution_limits)

    for panel in distribution_axes:
        panel.legend(fontsize=8)

    spanning_f = prediction_values(
        numerator_model, "predict", contour_spanning_dataset
    )
    spanning_g = prediction_values(
        numerator_model, "predict_secondary", contour_spanning_dataset
    )
    numerator_eta = prediction_values(
        numerator_model, "predict_eta", contour_spanning_dataset
    )
    denominator_eta = prediction_values(
        denominator_model, "predict_eta", contour_spanning_dataset
    )

    prediction_specs = {
        "f": (
            r"numerator $f(x)$",
            spanning_f,
            plot_colors["f"],
            prediction_linestyles["component"],
        ),
        "g": (
            r"numerator $g(x)$",
            spanning_g,
            plot_colors["g"],
            prediction_linestyles["component"],
        ),
        "numerator_eta": (
            r"numerator $\eta(x)$",
            numerator_eta,
            plot_colors["eta"],
            prediction_linestyles["component"],
        ),
        "numerator_eta_plus": (
            r"numerator $1+\eta(x)$",
            1.0 + numerator_eta,
            plot_colors["eta_plus"],
            prediction_linestyles["component"],
        ),
        "numerator_eta_minus": (
            r"numerator $1-\eta(x)$",
            1.0 - numerator_eta,
            plot_colors["eta_minus"],
            prediction_linestyles["component"],
        ),
        "f_eta_plus": (
            r"numerator $f(x)(1+\eta(x))$",
            spanning_f * (1.0 + numerator_eta),
            plot_colors["f"],
            prediction_linestyles["product"],
        ),
        "g_eta_minus": (
            r"numerator $g(x)(1-\eta(x))$",
            spanning_g * (1.0 - numerator_eta),
            plot_colors["g"],
            prediction_linestyles["product"],
        ),
        "denominator_eta": (
            r"denominator $\eta(x)$",
            denominator_eta,
            plot_colors["eta"],
            prediction_linestyles["denominator"],
        ),
        "denominator_eta_plus": (
            r"denominator $1+\eta(x)$",
            1.0 + denominator_eta,
            plot_colors["eta_plus"],
            prediction_linestyles["denominator"],
        ),
        "denominator_eta_minus": (
            r"denominator $1-\eta(x)$",
            1.0 - denominator_eta,
            plot_colors["eta_minus"],
            prediction_linestyles["denominator"],
        ),
    }
    sr_prediction_keys = tuple(prediction_specs)
    cr_prediction_keys = (
        "numerator_eta",
        "numerator_eta_plus",
        "numerator_eta_minus",
        "denominator_eta",
        "denominator_eta_plus",
        "denominator_eta_minus",
    )

    def project_prediction(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return utils__contour_model_prediction(
            prediction_function=lambda _: values,
            spanning_dataset=contour_spanning_dataset,
            along_observables=selected_observables,
            prediction_transform=np.asarray,
        )

    projected_predictions = {
        key: project_prediction(specification[1])
        for key, specification in prediction_specs.items()
    }
    finite_prediction_chunks = []
    for _, contour in projected_predictions.values():
        flattened_contour = utils__flatten_histogram_values(contour)
        finite_prediction_chunks.append(
            flattened_contour[np.isfinite(flattened_contour)]
        )
    finite_prediction_values = np.concatenate(finite_prediction_chunks)
    prediction_minimum = min(0.0, float(np.min(finite_prediction_values)))
    prediction_maximum = max(1.0, float(np.max(finite_prediction_values)))
    prediction_span = prediction_maximum - prediction_minimum
    prediction_padding = 0.05 * prediction_span if prediction_span > 0 else 0.5
    prediction_limits = (
        prediction_minimum - prediction_padding,
        prediction_maximum + prediction_padding,
    )

    def draw_prediction_panel(
        panel: plt.Axes,
        prediction_keys: Tuple[str, ...],
        region_name: str,
    ) -> None:
        first_coordinates, _ = projected_predictions[prediction_keys[0]]
        if ndim == 1:
            panel.axhline(0.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            panel.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            for prediction_key in prediction_keys:
                label, _, color, linestyle = prediction_specs[prediction_key]
                coordinates, contour = projected_predictions[prediction_key]
                panel.plot(
                    utils__flatten_histogram_values(coordinates),
                    utils__flatten_histogram_values(contour),
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                )
            panel.set_ylim(prediction_limits)
        else:
            x_values = np.unique(first_coordinates[:, 0])
            y_values = np.unique(first_coordinates[:, 1])
            prediction_xx, prediction_yy = np.meshgrid(
                x_values, y_values, indexing="ij"
            )
            for reference_value in (0.0, 1.0):
                panel.plot_surface(
                    prediction_xx,
                    prediction_yy,
                    np.full_like(prediction_xx, reference_value),
                    color="gray",
                    linewidth=0,
                    alpha=0.06,
                    shade=False,
                )
            for prediction_key in prediction_keys:
                label, _, color, linestyle = prediction_specs[prediction_key]
                coordinates, contour = projected_predictions[prediction_key]
                x_values = np.unique(coordinates[:, 0])
                y_values = np.unique(coordinates[:, 1])
                prediction_xx, prediction_yy = np.meshgrid(
                    x_values, y_values, indexing="ij"
                )
                contour_grid = np.asarray(contour).reshape(
                    len(x_values), len(y_values)
                )
                panel.plot_wireframe(
                    prediction_xx,
                    prediction_yy,
                    contour_grid,
                    color=color,
                    linestyle=linestyle,
                    linewidth=0.8,
                    alpha=0.9,
                )
                panel.plot([], [], [], label=label, color=color, linestyle=linestyle)
            panel.set_zlim(prediction_limits)

        set_observable_axes(panel, "model prediction")
        panel.set_title(f"{region_name} predictions")
        panel.legend(fontsize=7)

    draw_prediction_panel(sr_prediction_ax, sr_prediction_keys, "SR")
    draw_prediction_panel(cr_prediction_ax, cr_prediction_keys, "CR")

    return fig
