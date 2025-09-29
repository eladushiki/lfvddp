from pathlib import Path
from typing import List, Optional
from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetConfig, DatasetParameters, GeneratedDatasetParameters
from data_tools.detector.detector_config import DetectorConfig
from frame.aggregate import ResultAggregator
from frame.file_structure import CONTEXT_FILE_NAME
from neural_networks.utils import predict_sample_ndf_hypothesis_weights
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import patches
from plot.plotting_config import PlottingConfig
from plot.carpenter import Carpenter
from scipy.stats import chi2
from tensorflow.keras.models import Model  # type: ignore

from frame.context.execution_context import ExecutionContext
from plot.plot_utils import HandlerCircle, HandlerRect, utils__datset_histogram_sliced, utils__get_signal_dataset_parameters, utils__sample_over_background_histograms_sliced
from train.train_config import TrainConfig


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
    '''
    The funcion creates the plot of the evolution in the epochs of the [2.5%, 25%, 50%, 75%, 97.5%] quantiles of the toy sample distribution.
    The percentile lines for the target chi2 distribution are shown as a reference.
    
    patience:      (int) interval between two check points (epochs).
    tvalues_check: (numpy array shape (N_toys, N_check_points)) array of t=-2*loss
    df:            (int) chi2 degrees of freedom
    '''
    if not isinstance(config := context.config, PlottingConfig):
        raise ValueError(f"Expected context.config to be of type {PlottingConfig}, got {type(config)}")

    # Training results aggregation
    agg = ResultAggregator(Path(config.plot__target_run_parent_directory))
    all_model_t_test_statistics = agg.all_test_statistics
    epochs = agg.all_epochs

    # Framing
    c = Carpenter(context)
    fig  = c.figure()
    ax = fig.add_subplot(111)

    # Drawing
    legend = []
    quantiles   = [2.5, 25, 50, 75, 97.5]
    percentiles = np.apply_along_axis(lambda x: np.nanpercentile(x, quantiles), 0, all_model_t_test_statistics)
    colors = ['violet', 'hotpink', 'mediumvioletred', 'mediumorchid', 'darkviolet']
    
    # Training percentile progression
    for j in range(percentiles.shape[0]):
        plt.plot(epochs, percentiles[j, :], linewidth=3, color=colors[j])
        legend.append(str(quantiles[j])+'% quantile')
    
    # chi2 reference
    for j in range(percentiles.shape[0]):
        plt.plot(epochs, chi2.ppf(quantiles[j] / 100., df=config.train__nn_degrees_of_freedom, loc=0, scale=1)*np.ones_like(epochs),
                color=colors[j], ls='--', linewidth=1)
        if j==0: legend.append("Target "+r"$\chi^2($"+str(config.train__nn_degrees_of_freedom)+")")

    # Labeling
    plt.title(r"$\chi^2$ percentile progression", fontsize=24)

    if np.any(np.isnan(all_model_t_test_statistics)):
        legend.append(f"Nan percent: {np.count_nonzero(np.isnan(all_model_t_test_statistics))/all_model_t_test_statistics.size*100:.2f}")
    plt.legend(legend, frameon=False, markerscale=0)
    
    plt.xlabel('Training Epochs', fontsize=22)
    plt.ylabel('t', fontsize=22)
    plt.xlim(0, np.max(epochs))
    plt.ylim(0, np.nanmax(percentiles))
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.ticklabel_format(axis="x", style="scientific", scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(18)
    
    return fig


def t_distribution_plot(
        context: ExecutionContext,
        number_of_bins: int,
    ) -> Figure:
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution. 
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    '''
    if not isinstance(config := context.config, PlottingConfig):
        raise ValueError(f"Expected context.config to be of type {PlottingConfig}, got {type(config)}")
    if not isinstance(config, TrainConfig):
        raise ValueError(f"Expected context.config to be of type {TrainConfig}, got {type(config)}")
    style = config.plot__figure_styling["plot"]

    # Figure
    c = Carpenter(context)
    fig  = c.figure()
    ax = fig.add_subplot(111)

    agg = ResultAggregator(Path(config.plot__target_run_parent_directory))
    t = agg.all_t_values

    # Limits
    chi2_begin = chi2.ppf(0.0001, chi2_dof := config.train__nn_degrees_of_freedom)
    chi2_end = chi2.ppf(0.9999, chi2_dof)
    xmin = max([min([np.min(t), chi2_begin]), 0])
    xmax = max([np.percentile(t, 95), chi2_end])

    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, number_of_bins + 1)
    bin_width = (xmax - xmin) * 1./number_of_bins
    label     = f"median: {str(np.around(np.median(t), 2))} \n" \
                f"std: {str(np.around(np.std(t), 2))}"
    if (n_nans := agg.nan_t_values) > 0:
        label += f"\nDid not converge: {n_nans / t.size * 100:.2f}%"
        
    h = ax.hist(
        t,
        weights=np.ones_like(t)*1./(t.shape[0]*bin_width),
        color=style["histogram_color"],
        ec=style["edge_color"],
        bins=bins,
        label=label,
    )
    
    y_error     = np.sqrt(h[0] / (t.shape[0] * bin_width))
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    ax.errorbar(
        bin_centers,
        h[0],
        yerr=y_error,
        color=style["edge_color"],
        marker='o', 
        ls='',
    )

    # plot reference chi2
    bin_centers  = np.linspace(
        chi2_begin,
        chi2_end,
        1000
    )

    ax.plot(
        bin_centers,
        chi2.pdf(bin_centers, chi2_dof),
        style["chi2_color"],
        linewidth=style["linewidth"],
        alpha=style["alpha"],
        label=f'$\chi^{2}_{{{chi2_dof}}}$',
    )

    # Legend
    circ = patches.Circle((0,0), 1, facecolor=style["histogram_color"], edgecolor=style["edge_color"])
    rect1 = patches.Rectangle((0,0), 1, 1, color=style["chi2_color"], alpha=style["alpha"])
    
    ax.legend(
        (circ, rect1),
        (label, f'$\chi^{2}_{{{config.train__nn_degrees_of_freedom}}}$'),
        handler_map={
            patches.Rectangle: HandlerRect(),
            patches.Circle: HandlerCircle(),
        },
        frameon=False,
    )
    
    # Texting
    histogram_title = f"Distribution of t values over {len(t)} test runs"
    ax.set_title(histogram_title, fontsize=30, pad=20)
    ax.set_xlabel('t', fontsize=22, labelpad=20)
    ax.set_ylabel('Bin Probability', fontsize=22, labelpad=20)
    ax.set_ylim(0, 0.1)
    plt.yticks([0.03, 0.06, 0.09])
    plt.xticks()

    return fig


def performance_plot(
        context: ExecutionContext,
        background_only_t_values_parent_directory: str,
        signal_t_values_parent_directories: List[str],
    ):
    '''
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
    '''
    if not isinstance(plot_config := context.config, PlottingConfig):
        raise ValueError(f"Expected context.config to be of type {PlottingConfig}, got {type(plot_config)}")

    # Validate background configuration
    ## this has to be a generated type, else the distribution is not well known
    background_context = ExecutionContext.naive_load_from_file(Path(background_only_t_values_parent_directory) / CONTEXT_FILE_NAME)
    background_config: DatasetConfig = background_context.config
    for background_dataset_name in background_config._dataset__names:
        background_dataset_properties: DatasetParameters = background_config._dataset__parameters(background_dataset_name)
        assert isinstance(background_dataset_properties, GeneratedDatasetParameters), \
            f"performance plot possible only for generated datasets, got {background_dataset_properties.type}"
        assert background_dataset_properties.dataset__number_of_signal_events == 0, \
            f"background dataset expected to have only background events, {background_dataset_name} has {background_dataset_properties.dataset__number_of_signal_events} signal events"

    # Gather background data
    background_agg = ResultAggregator(Path(background_only_t_values_parent_directory))
    background_t_dist = background_agg.all_t_values

    # Result lists
    ## The analytic calculation of significance based on input parameters, by eq. (33) in the last paper
    mean_injected_significances = []
    injected_significance_stds = []

    ## The significance by the observed chance to generate an equal or larger t value had this been a 
    ## background only dataset, and confidence bounds
    observed_significances = []
    observed_significances_upper_confidence_bounds = []
    observed_significances_lower_confidence_bounds = []
    observed_significances_by_gaussian_fit = []

    for signal_t_values_dir in signal_t_values_parent_directories:
        
        # Load corresponding dataset
        signal_context = ExecutionContext.naive_load_from_file(Path(signal_t_values_dir) / CONTEXT_FILE_NAME)
        signal_dataset_parameters = utils__get_signal_dataset_parameters(signal_context)

        # Gather data
        signal_agg = ResultAggregator(Path(signal_t_values_dir))
        signal_t_dist = signal_agg.all_t_values

        # Calculate the injected significance centers using the mean number of events.
        # Those are before introducting poisson fluctuations.
        mean_injected_significances.append(calc_injected_t_significance_by_sqrt_q0_continuous(
            background_pdf=signal_dataset_parameters.dataset_generated__background_pdf,
            signal_pdf=signal_dataset_parameters.dataset_generated__signal_pdf,
            n_background_events=signal_dataset_parameters.dataset__mean_number_of_background_events,
            n_signal_events=signal_dataset_parameters.dataset__mean_number_of_signal_events,
            upper_limit=max(signal_t_dist.max(), background_t_dist.max()),
        ))
        injected_significance_stds.append(np.std(
            signal_agg.all_injected_significances
        ))

        # Calculate observed significance and +-1 sigma confidence interval
        observed_significances.append(
            calc_median_t_significance_relative_to_background(
                background_t_dist,
                signal_t_dist,
        ))
        signal_t_dist_std = np.std(signal_t_dist)
        observed_significances_lower_confidence_bounds.append(
            calc_t_significance_relative_to_background(
                np.mean(signal_t_dist) - signal_t_dist_std, background_t_dist
        ))
        observed_significances_upper_confidence_bounds.append(
            calc_t_significance_relative_to_background(
                np.mean(signal_t_dist) + signal_t_dist_std, background_t_dist
        ))
        observed_significances_by_gaussian_fit.append(
            calc_t_significance_by_gaussian_fit_percentile(
                background_only_distribution=background_t_dist,
                t_value=np.median(signal_t_dist),
        ))

    # Sort all results by injected significance    
    sort = np.argsort(np.array(mean_injected_significances))
    mean_injected_significances = np.array(mean_injected_significances)[sort]
    injected_significance_stds = np.array(injected_significance_stds)[sort]
    observed_significances = np.array(observed_significances)[sort]
    observed_significances_lower_confidence_bounds = np.array(observed_significances_lower_confidence_bounds)[sort]
    observed_significances_upper_confidence_bounds = np.array(observed_significances_upper_confidence_bounds)[sort]
    observed_significances_by_gaussian_fit = np.array(observed_significances_by_gaussian_fit)[sort]

    # Framing
    c = Carpenter(context)
    fig  = c.figure()
    ax = fig.add_subplot(111)

    # Borders
    graph_border = 1
    clean_y_significances = np.concatenate([
        observed_significances[np.isfinite(observed_significances)],
        observed_significances_lower_confidence_bounds[np.isfinite(observed_significances_lower_confidence_bounds)],
        observed_significances_upper_confidence_bounds[np.isfinite(observed_significances_upper_confidence_bounds)],
        observed_significances_by_gaussian_fit[np.isfinite(observed_significances_by_gaussian_fit)],
    ])
    min_x = max(min(mean_injected_significances) - graph_border, 0)
    max_x = max(mean_injected_significances) + graph_border
    min_y = max(min(clean_y_significances) - graph_border, 0)
    max_y = max(clean_y_significances) + graph_border
    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_y,max_y)

    # Plots
    colors = plt.get_cmap('cool')
    
    ax.plot(mean_injected_significances, observed_significances_by_gaussian_fit, color=colors(0.75), linewidth=2, linestyle='--', label="gaussian fit significance")
    ax.plot(mean_injected_significances, observed_significances, color=colors(0.5), label="observed significance", linewidth=2)
    ax.fill_between(
        mean_injected_significances,
        np.clip(observed_significances_lower_confidence_bounds, a_min=0, a_max=max_y),
        np.clip(observed_significances_upper_confidence_bounds, a_min=0, a_max=max_y),
        color=colors(1),
        linewidth=2,
        alpha=0.1
    )

    # Error bars
    ax.errorbar(
        mean_injected_significances,
        observed_significances,
        xerr=injected_significance_stds,
    )
    
    # Texting
    ax.set_xlabel(r'injected $\sqrt{q_0}$', fontsize=21)
    ax.set_ylabel('measured significance', fontsize=21)
    ax.set_title("measured vs injected signal significance", fontsize=24)
    legend = ax.legend(loc='lower right', fontsize=20, fancybox=True, frameon=False)

    # Styling
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0.0)
    ax.tick_params(labelsize=20)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='lower'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='lower'))

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
    bins, _ = context.config.observable_bins(observable or context.config.detector__detect_observable_names[0])

    datasets = sample_hollow_datasets + background_solid_datasets
    legends = sample_legends + background_legends
    ax = fig.add_subplot(111)
    for i, background in enumerate(datasets):
        utils__datset_histogram_sliced(
            ax=ax,
            bins=bins,
            dataset=background,
            along_observable=observable,
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
        along_observable=observable,
        label="original sample",
        histtype="stepfilled",
    )
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=processed_sample,
        alternative_weights=np.ones(shape=(processed_sample.n_samples, 1)),
        along_observable=observable,
        label="detector affected sample",
        histtype="stepfilled",
    )
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=processed_sample,
        # the usual weights
        along_observable=observable,
        label="detector affected sample (weight adjusted)",
        histtype="step",
    )

    ax.set_title("Sample Generation Process Illustration", fontsize=24)
    ax.set_xlabel(f"{observable}", fontsize=20)
    ax.set_ylabel("number of events", fontsize=20)
    ax.legend()
    return fig


def plot_prediction_process_sliced(
        context: ExecutionContext,
        experiment_sample: DataSet,
        reference_sample: DataSet,
        trained_tau_model: Model,
        trained_delta_model: Optional[Model],
        title="Datasets Along the Process",
        along_observable: Optional[str] = None,
        sample_legend="training sample (det. reconstructed)",
        background_legend="reference sample (det. reconstructed)",
        tau_prediction_legend="tau model prediction",
        delta_prediction_legend="delta model prediction",
        tau_prediction_color="cyan",
        delta_prediction_color="magenta",
        xlabel: str = "mass",
        ylabel: str = "number of events",
    ):
    """
    Give a single histogram featuring:
    - raw experimental data sample
    - experimental data sample weighterd to compensate detector losses
    - weighted reference sample
    - tau model prediction for the reconstruction of the experimental data
    - delta model prediction for the reconstruction of the experimental data (if provided)
    """
    if not isinstance((config := context.config), TrainConfig):
        raise ValueError("The context config is not a TrainConfig.")
    if not isinstance(config, DatasetConfig):
        raise ValueError("The context config is not a DatasetConfig.")
    if not isinstance(config, DetectorConfig):
        raise ValueError("The context config is not a DetectorConfig.")

    if along_observable is None:
        along_observable = config.detector__detect_observable_names[0]
    
    c = Carpenter(context)
    fig = c.figure()
    ax = fig.add_subplot(111)

    bins, bin_centers = config.observable_bins(along_observable)

    utils__sample_over_background_histograms_sliced(
        ax=ax,
        sample=experiment_sample,
        background=reference_sample,
        bins=bins,
        along_observable=along_observable,
        sample_legend=sample_legend,
        background_legend=background_legend,
    )

    prediction_hist_kwargs = {
        "histtype": "step",
        "log": True,
        "lw": 0,
    }
    prediction_scatter_kwargs = {
        "s": 30,
        "edgecolor": "black",
    }

    _reference_data = reference_sample.slice_along_observable_names(along_observable)
    tau_hypothesis_weights = predict_sample_ndf_hypothesis_weights(trained_model=trained_tau_model, predicted_distribution_corrected_size=experiment_sample.corrected_n_samples, reference_ndf_estimation=reference_sample)
    predicted_tau_ndf = plt.hist(_reference_data, weights=tau_hypothesis_weights, bins=list(bins), **prediction_hist_kwargs)
    ax.scatter(bin_centers, predicted_tau_ndf[0], label=tau_prediction_legend, color=tau_prediction_color, **prediction_scatter_kwargs)

    if trained_delta_model is not None:
        delta_hypothesis_weights = predict_sample_ndf_hypothesis_weights(trained_model=trained_delta_model, predicted_distribution_corrected_size=experiment_sample.corrected_n_samples, reference_ndf_estimation=reference_sample)
        predicted_delta_ndf = plt.hist(_reference_data, weights=delta_hypothesis_weights, bins=list(bins), **prediction_hist_kwargs)
        ax.scatter(bin_centers, predicted_delta_ndf[0], label=delta_prediction_legend, color=delta_prediction_color, **prediction_scatter_kwargs)

    ax.set_title(title, fontsize=30, pad=20)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig
