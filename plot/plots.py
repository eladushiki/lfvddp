from pathlib import Path
from typing import List, Optional, Union
from data_tools.data_utils import DataSet, create_slice_containing_bins
from data_tools.dataset_config import DatasetConfig, DatasetParameters, GeneratedDatasetParameters
from data_tools.profile_likelihood import calc_t_significance_by_chi2_percentile, calc_median_t_significance_relative_to_background, calc_t_significance_relative_to_background, calc_injected_t_significance_by_sqrt_q0_continuous
from frame.aggregate import ResultAggregator
from frame.file_structure import CONTEXT_FILE_NAME
from neural_networks.NPLM_adapters import predict_sample_ndf_hypothesis_weights
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager, patches
import plot
from plot.plotting_config import PlottingConfig
from plot.carpenter import Carpenter
from scipy.stats import chi2,norm
from tensorflow.keras.models import Model  # type: ignore
from matplotlib.animation import FuncAnimation
import scipy.special as spc
from IPython.display import HTML

from frame.context.execution_context import ExecutionContext
from plot.plot_utils import HandlerCircle, HandlerRect, utils__datset_histogram_sliced, utils__sample_over_background_histograms_sliced, em_results, utils__create_slice_containing_bins, get_z_score, results, scientific_number
from train.train_config import TrainConfig


# DEVELOPER NOTE: Each function here can ba called from "PlottingConfig" BY NAME.
# Implement any new plot function here, and you will be able to call it automatically.
# This being said, the format for implementation has to be:
#
# def <name from plot_config.json>(context: ExecutionContext, <instructions from plot_config.json>) -> matplotlib.figure.Figure:
#    ...
#
# Should not save the figure by itself!!! It is done in a well documented way in the calling function.


def Plot_Percentiles_ref(
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


def plot_old_t_distribution(
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
    
    _, legend_labels = ax.get_legend_handles_labels()
    legend_labels.append(f"Did not converge: {np.sum(t > 0) / t.size * 100:.2f}%")
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
    the injected z = sqrt(q0) for the exponential distribution
    with a given signal type.

    Data needed to generate the plot:
    - t values distribution for a run with background only.
        contained in a single directory and is used as a
        reference for all signal distributions.
    - A set of t values distributions, each with a different
        injected signal strength. Parameters of each are picked
        from the context file, from the data specification under
        the corresponding signal dataset name BY ORDER.

    The plot__target_run_parent_directory has no significance to
    not cause ambiguity.
    '''
    if not isinstance(plot_config := context.config, PlottingConfig):
        raise ValueError(f"Expected context.config to be of type {PlottingConfig}, got {type(plot_config)}")

    # Validate background configuration
    ## this has to be a generated type, else the distribution is not well known
    background_context = ExecutionContext.naive_load_from_file(Path(background_only_t_values_parent_directory) / CONTEXT_FILE_NAME)
    mean_number_of_background_events = 0
    background_config: DatasetConfig = background_context.config
    for background_dataset_name in background_config._dataset__names:
        background_dataset_properties: DatasetParameters = background_config._dataset__parameters(background_dataset_name)
        assert isinstance(background_dataset_properties, GeneratedDatasetParameters), \
            f"performance plot possible only for generated datasets, got {background_dataset_properties.type}"
        assert background_dataset_properties.dataset__number_of_signal_events == 0, \
            f"background dataset expected to have only background events, {background_dataset_name} has {background_dataset_properties.dataset__number_of_signal_events} signal events"
        mean_number_of_background_events = background_dataset_properties.dataset__mean_number_of_background_events  # Assuming all datasets have the same mean number of events

    # Gather background data
    background_agg = ResultAggregator(Path(background_only_t_values_parent_directory))
    background_t_dist = background_agg.all_t_values

    # Result lists
    observed_significances = []
    chi2_significances = []
    injected_significances = []
    observed_significance_upper_confidence_bounds = []
    observed_significances_lower_confidence_bounds = []

    for signal_t_values_dir in signal_t_values_parent_directories:
        signal_context = ExecutionContext.naive_load_from_file(Path(signal_t_values_dir) / CONTEXT_FILE_NAME)

        # Validate signal configuration
        mean_number_of_signal_events = 0
        signal_config: Union[DatasetConfig, TrainConfig] = signal_context.config
        for dataset_name in signal_config._dataset__names:
            dataset_properties: DatasetParameters = signal_config._dataset__parameters(dataset_name)
            assert isinstance(dataset_properties, GeneratedDatasetParameters), \
                f"performance plot possible only for generated datasets, got {dataset_properties.type}"

            # We do assume there is a signal in only one dataset
            if (current_mean_number_of_signal_events := dataset_properties.dataset__mean_number_of_signal_events) != 0:
                assert mean_number_of_signal_events == 0, \
                    f"multiple signal datasets found, {dataset_name} being the second"
                mean_number_of_signal_events = current_mean_number_of_signal_events
                signal_dataset_properties = dataset_properties

        assert mean_number_of_signal_events != 0, \
            f"No dataset with signal events found among {signal_config._dataset__names}"

        # Gather data
        signal_agg = ResultAggregator(Path(signal_t_values_dir))
        signal_t_dist = signal_agg.all_t_values

        chi2_significances.append(calc_t_significance_by_chi2_percentile(
            t_distribution=signal_t_dist,
            degrees_of_freedom=signal_config.train__nn_degrees_of_freedom,
        ))
        
        injected_significances.append(calc_injected_t_significance_by_sqrt_q0_continuous(
            background_pdf=signal_dataset_properties.dataset__background_pdf,
            signal_pdf=signal_dataset_properties.dataset__signal_pdf,
            n_background_events=mean_number_of_background_events,  # The mean numbers are the theoretic ones, before injecting poisson error
            n_signal_events=mean_number_of_signal_events,
            upper_limit=max(signal_agg.all_t_values.max(), background_agg.all_t_values.max()),
        ))

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
        observed_significance_upper_confidence_bounds.append(
            calc_t_significance_relative_to_background(
                np.mean(signal_t_dist) + signal_t_dist_std, background_t_dist
        ))
    
    sort = np.argsort(np.array(injected_significances))
    # Formerly approx_z_score, the chi2 percentile the median is at:
    chi2_significances = np.array(chi2_significances)[sort]
    # Formerly Sig_q0, analytic integral of generating functions:
    injected_significances = np.array(injected_significances)[sort]
    # Formerly Sig_z_score, percentile of mean signal t in background t distribution:
    observed_significances = np.array(observed_significances)[sort]
    observed_significances_lower_confidence_bounds = np.array(observed_significances_lower_confidence_bounds)[sort]
    observed_significance_upper_confidence_bounds = np.array(observed_significance_upper_confidence_bounds)[sort]
    
    # Framing
    c = Carpenter(context)
    fig  = c.figure()
    ax = fig.add_subplot(111)

    # Borders
    graph_border = 1
    clean_y_significances = np.concatenate([
        chi2_significances[np.isfinite(chi2_significances)],
        observed_significances[np.isfinite(observed_significances)],
        observed_significances_lower_confidence_bounds[np.isfinite(observed_significances_lower_confidence_bounds)],
        observed_significance_upper_confidence_bounds[np.isfinite(observed_significance_upper_confidence_bounds)],
    ])
    min_x = max(min(injected_significances) - graph_border, 0)
    max_x = max(injected_significances) + graph_border
    min_y = max(min(clean_y_significances) - graph_border, 0)
    max_y = max(clean_y_significances) + graph_border
    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_y,max_y)

    # Plots
    colors = plt.get_cmap('cool')
    
    ax.plot(injected_significances, chi2_significances, color=colors(0), linewidth=2, linestyle='--')
    ax.plot(injected_significances, observed_significances, color=colors(0.5), label="observed significance", linewidth=2)
    ax.fill_between(
        injected_significances,
        observed_significances_lower_confidence_bounds,
        np.clip(observed_significance_upper_confidence_bounds, a_max=max_y),
        color=colors(1),
        linewidth=2,
        alpha=0.1
    )
    chi2_label = r"$\chi^2_" + str(background_config.train__nn_degrees_of_freedom) + r"$"
    chi2_curve = mpl.lines.Line2D([], [], color='black', linestyle='--', label=chi2_label)
    
    # Texting
    ax.set_xlabel(r'injected $\sqrt{q_0}$', fontsize=21)
    ax.set_ylabel('measured significance', fontsize=21)
    ax.set_title("measured vs injected signal significance", fontsize=24)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(chi2_curve)
    legend = ax.legend(
        handles=handles,
        labels=labels + [chi2_label],
        loc='lower right',
        fontsize=20,
        fancybox=True,
        frameon=False
    )

    # Styling
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0.0)
    ax.tick_params(labelsize=20)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='lower'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, prune='lower'))

    return fig


def em_luminosity_plot(Bkg_only_files, S_in_20=1200,title="", title_fs=24, labels_fs=21, ticks_fs=20, legend_fs=20, save=False, save_path='', saved_file_name=''):
    '''
    The function creates a plot of the measured significance as a function of the luminosity (in inverse fb) for the em distribution.

    S_in_20:        (int) number of signal events injected to the 20 fb^-1 background (1:2-1:2).

    For more details, see em_performance_plot's docstring.
    '''
    raise NotImplementedError('This function is not implemented yet.')
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(223)
    fig.set_facecolor('white')
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    colors = plt.get_cmap('spring')
    luminosity_vec = []
    Sig_z_score = []
    approx_z_score = []
    Sig_q0 = []
    for i,Bkg_only_file in enumerate(Bkg_only_files):
        if Bkg_only_file.Bkg_sample == 'exp':
            print('Used em plot function for exp sample.')
            return
        luminosity = 20*Bkg_only_file.Bkg_ratio
        Bkg_only_file_name = Bkg_only_file.file
        Sig_file_names = Bkg_only_file.get_signal_files()
        
        N_sig = int(S_in_20*2*Bkg_only_file.Bkg_ratio)
        print(N_sig)
        for sig in Sig_file_names:
            Sig_file = em_results(sig)
            if Sig_file.Sig_events==N_sig:
                print(Sig_file.Sig_events)
                z_score, Sig_t, Bkg_t = get_z_score(Sig_file,Bkg_only_file)
                Sig_z_score.append(z_score)
                approx_z_score.append(norm.ppf(chi2.cdf(np.median(Sig_t), df=12)))
                Sig_q0.append(Sig_file.get_sqrt_q0())
                luminosity_vec.append(luminosity)

    if len(luminosity_vec)>0:
        luminosity_vec = np.array(luminosity_vec)
        Sig_z_score = np.array(Sig_z_score)
        approx_z_score = np.array(approx_z_score)
        Sig_q0 = np.array(Sig_q0)
        sort = np.argsort(luminosity_vec)
        luminosity_vec = luminosity_vec[sort]
        Sig_z_score = Sig_z_score[sort]
        approx_z_score = approx_z_score[sort]
        Sig_q0 = Sig_q0[sort]
        ax.plot(luminosity_vec, Sig_z_score, color=colors[1], label=r"$Z_{\rm meas}$", linewidth=2)
        ax.plot(luminosity_vec,Sig_q0, color=colors[2], label=r"$\sqrt{q_0}$", linewidth=2,ls = ':')
        ax.plot(luminosity_vec, approx_z_score, color=colors[3],label = r"$\chi^{2}_{12}$",linewidth=2, linestyle='--')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_xlim(left=5)
        ax.set_xlabel(r'Luminosity $[{fb}^{-1}]$', fontsize=labels_fs)
        ax.set_ylabel('Z', fontsize=labels_fs)
        ax.set_title(title, fontsize=title_fs)
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend( labels=labels, loc='upper left', fontsize=legend_fs, fancybox=True, frameon=False)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(1)
        legend.get_frame().set_linewidth(0.2)
        ax.tick_params(labelsize=ticks_fs)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
        xmin, xmax = ax.get_xlim()
        ax.set_xticks(ticks=np.arange(xmin, xmax+1, 5))
        if save:
            if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
            else:
                plt.savefig(save_path+saved_file_name+'.pdf')
        plt.show()


def animated_t_distribution(results_file:results, df, epoch = 500000,xmin=0, xmax=300, nbins=10, samples_to_take='all', frames=300, repeat=False, label='', title='', save=False, save_path='', file_name=''):
    '''
    Creates an animated gif of the t distribution.

    frames:         (int) number of frames in the animation.
    repeat:         (bool) if True, the animation repeats itself.
                    If repeat=False is required, other file formats should be used (instead of gif).
    For more details, see t_distribution's docstring.
    '''
    raise NotImplementedError('This function is not implemented yet.')
    t_dict = results_file.get_t_history_dict()
    max_epoch = max(t_dict.keys())
    t = t_dict[epoch] if epoch in t_dict.keys() else t_dict[max_epoch]
    if samples_to_take != 'all':
        t = t[:samples_to_take]
    Ref_events = results_file.Ref_events
    Bkg_events = results_file.Bkg_events
    Sig_events = results_file._config.train__signal_number_of_events
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    box = ax.get_position()
    fig.patch.set_facecolor('white')
    bins      = np.linspace(xmin, xmax, nbins+1)
    Z_obs     = norm.ppf(chi2.cdf(np.median(t), df))
    t_obs_err = 1.2533*np.std(t)*1./np.sqrt(t.shape[0])
    Z_obs_p   = norm.ppf(chi2.cdf(np.median(t)+t_obs_err, df))
    Z_obs_m   = norm.ppf(chi2.cdf(np.median(t)-t_obs_err, df))
    t_median = np.median(t[np.where(np.logical_not(np.isnan(t)))])
    t_std = np.std(t[np.where(np.logical_not(np.isnan(t)))])
    t_num_of_nan = np.sum(np.isnan(t))
    if label =='':
        label  = 'med: %s \nstd: %s'%(str(np.around(t_median, 2)), str(np.around(t_std, 2)))
    if title == '':
        title = r'$N_A^0=$'+f"{scientific_number(Bkg_events)}"+r',   $N_B^0=$'+f"{scientific_number(Ref_events)}"
        if Sig_events > 0:
            title += r',   $N_{sig}^0=$'+f"{scientific_number(Sig_events)}"
    binswidth = (xmax-xmin)*1./nbins
    if results_file.NPLM == 'False':
        color = 'plum'
        ec = 'darkorchid'
        chi2_color = 'grey'
    elif results_file.NPLM == 'True':
        color = 'lightcoral'
        ec = 'red'
        chi2_color = 'grey'
    h = ax.hist(t, weights=np.ones_like(t)*1./(t.shape[0]*binswidth), color=color, ec=ec,
                 bins=bins, label=label)
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 1000)
    ax.plot(x, chi2.pdf(x, df), color=chi2_color, lw=5, alpha=0.8, label=f'$\chi^{2}_{{{df}}}$')
    font = font_manager.FontProperties(family='serif', size=24) 
    circ = patches.Circle((0,0), 1, facecolor=color, edgecolor=ec)
    rect1 = patches.Rectangle((0,0), 1, 1, color=chi2_color, alpha=0.8)
    legend = ax.legend((circ, rect1), (label, f'$\chi^{2}_{{{df}}}$'),
            handler_map={
            patches.Rectangle: HandlerRect(),
            patches.Circle: HandlerCircle(),
            },
            prop=font,frameon=False)
    if t_num_of_nan > 0:
        rect2 = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        legend = ax.legend((circ, rect1, rect2), (label, f'$\chi^{2}_{{{df}}}$',f'NaN: {t_num_of_nan/t.shape[0]*100:.1f}%'),
            handler_map={
            patches.Rectangle: HandlerRect(),
            patches.Circle: HandlerCircle(),
            },
            prop=font,frameon=False)
    ax.set_xlabel('t', fontsize=24, fontname="serif", labelpad=20)
    ax.set_ylabel('PDF', fontsize=24, fontname="serif", labelpad=20)
    ax.set_ylim(0,0.1)
    plt.yticks([0.03,0.06,0.09], fontsize=24, fontname="serif")
    plt.xticks(fontsize=24, fontname="serif")
    ax.set_title(title, fontsize=30, fontname="serif", pad=20)
    mpl.rcParams['animation.embed_limit'] = 2**128

    def update(frame, h):
        for bar, real_height in zip(h[-1], h[0]):
            if frame == 0:
                bar.set_height(0)
            elif frame <= frames//2:
                bar.set_height(real_height * frame / (frames//2))
            else:
                amplitude = real_height * (frames - frame) / (2*frames)
                oscillations = amplitude * np.sin(frame / (frames//2) * np.pi)
                bar.set_height(real_height + oscillations)
                err = np.sqrt(h[0]/(t.shape[0]*binswidth))
                x   = 0.5*(bins[1:]+bins[:-1])
                if frame == frames-1: ax.errorbar(x, h[0], yerr = err, color='darkorchid', marker='o', ls='', lw=0.05, mew=0.1, animated=True)
        return h[-1]
    
    anim = FuncAnimation(fig, update, frames=frames, fargs=(h,), interval=20, blit=True, save_count=frames, repeat=repeat)
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = '1distribution'
            else: file_name += '_1distribution'
            anim.save(save_path+file_name+'.gif', writer='imagemagick', fps=30)
    return HTML(anim.to_jshtml())
    

def animated_t_2distributions(results_file1:results, results_file2:results, df, epoch = 500000,xmin=0, xmax=300, nbins=10, frames=300, repeat=False, label='', title='', save=False, save_path='', file_name=''):
    '''
    Creates an animated gif of the two t distributions for comaprison.
    For more details, see t_2distributions's and animated_t_distribution's docstrings.
    '''
    raise NotImplementedError('This function is not implemented yet.')

    t1_dict = results_file1.get_t_history_dict()
    t2_dict = results_file2.get_t_history_dict()
    max_epoch = min(max(t1_dict.keys()),max(t2_dict.keys()))
    t1 = t1_dict[epoch] if ((epoch in t1_dict.keys()) and (epoch in t2_dict.keys())) else t1_dict[max_epoch]
    t2 = t2_dict[epoch] if ((epoch in t1_dict.keys()) and (epoch in t2_dict.keys())) else t1_dict[max_epoch]

    color1 = ['plum', 'darkorchid']
    color2 = ['lightcoral', 'crimson']
    alpha = [0.8, 0.5]
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(12,9))
    fig.patch.set_facecolor('white')
    for i in [2,1]:
        t = t1 if i==1 else t2
        color = color1 if i==1 else color2
        bins      = np.linspace(xmin, xmax, nbins+1)
        Z_obs     = norm.ppf(chi2.cdf(np.median(t), df))
        t_obs_err = 1.2533*np.std(t)*1./np.sqrt(t.shape[0])
        Z_obs_p   = norm.ppf(chi2.cdf(np.median(t)+t_obs_err, df))
        Z_obs_m   = norm.ppf(chi2.cdf(np.median(t)-t_obs_err, df))
        label  = 'sample: %s\nsize: %i \nmedian: %s, std: %s\n'%(label, t.shape[0], str(np.around(np.median(t), 2)),str(np.around(np.std(t), 2)))
        label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
        binswidth = (xmax-xmin)*1./nbins
        h = plt.hist(t, weights=np.ones_like(t)*1./(t.shape[0]*binswidth), color=color[0], ec=color[1],
                    bins=bins, label=label, alpha=alpha[i-1])
        err = np.sqrt(h[0]/(t.shape[0]*binswidth))
        x   = 0.5*(bins[1:]+bins[:-1])
        if i == 2: plt.errorbar(x, h[0], yerr = err, color=color[1], marker='o', ls='')
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 1000)
    plt.plot(x, chi2.pdf(x, df),'rebeccapurple', lw=5, alpha=0.8, label=f'$\chi^{2}_{{{df}}}$')
    font = font_manager.FontProperties(family='serif', size=14) 
    plt.legend(prop=font,frameon=False)
    plt.xlabel('t', fontsize=18, fontname="serif")
    plt.ylabel('PDF', fontsize=18, fontname="serif")
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    plt.title(title, fontsize=18, fontname="serif")
    mpl.rcParams['animation.embed_limit'] = 2**128
    def update(frame, h):
        for bar, real_height in zip(h[-1], h[0]):
            if frame == 0:
                bar.set_height(0)
            elif frame <= frames//2:
                bar.set_height(real_height * frame / (frames//2))
            else:
                amplitude = real_height * (frames - frame) / (2*frames)
                oscillations = amplitude * np.sin(frame / (frames//2) * np.pi)
                bar.set_height(real_height + oscillations)
                err = np.sqrt(h[0]/(t.shape[0]*binswidth))
                x   = 0.5*(bins[1:]+bins[:-1])
                if frame == frames-1: plt.errorbar(x, h[0], yerr = err, color='darkorchid', marker='o', ls='', lw=0.05, mew=0.1)
        return h[-1]
    
    anim = FuncAnimation(fig, update, frames=frames, fargs=(h,), interval=20, blit=True, save_count=frames, repeat=repeat)
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = '2distributions'
            else: file_name += '_2distributions'
            anim.save(save_path+file_name+'.gif', writer='imagemagick', fps=30)
    return HTML(anim.to_jshtml())


def plot_samples_over_background_sliced(
        context: ExecutionContext,
        background_solid_datasets: List[DataSet] = [],
        sample_hollow_datasets: List[DataSet] = [],
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
    bins, _ = utils__create_slice_containing_bins(
        background_solid_datasets + sample_hollow_datasets,
    )

    datasets = sample_hollow_datasets + background_solid_datasets
    legends = sample_legends + background_legends
    ax = fig.add_subplot(111)
    for i, background in enumerate(datasets):
        utils__datset_histogram_sliced(
            ax=ax,
            bins=bins,
            dataset=background,
            label=legends[i],
            histtype="stepfilled" if i < len(sample_hollow_datasets) else "step",
        )
    ax.set_title(title)

    return fig


def plot_data_generation_sliced(
        context: ExecutionContext,
        original_sample: DataSet,
        processed_sample: DataSet,
):
    c = Carpenter(context)
    fig = c.figure()
    ax = fig.add_subplot(111)

    bins, _ = utils__create_slice_containing_bins([processed_sample])

    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=original_sample,
        # the usual weights
        histtype="stepfilled",
        label="original sample",
    )
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=processed_sample,
        alternative_weights=np.ones_like(processed_sample._data),
        histtype="stepfilled",
        label="detector affected sample",
    )
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=processed_sample,
        # the usual weights
        histtype="step",
        label="detector affected sample (weighted)",
    )

    ax.set_title("Data generation process")
    ax.legend()
    return fig


def plot_prediction_process_sliced(
        context: ExecutionContext,
        experiment_sample: DataSet,
        reference_sample: DataSet,
        trained_tau_model: Model,
        trained_delta_model: Optional[Model],
        title="Datasets Along the Process",
        along_dimension: int = 0,
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
    
    c = Carpenter(context)
    fig = c.figure()
    ax = fig.add_subplot(111)

    bins, bin_centers = utils__create_slice_containing_bins([experiment_sample, reference_sample])

    utils__sample_over_background_histograms_sliced(
        ax=ax,
        sample=experiment_sample,
        background=reference_sample,
        bins=bins,
        along_dimension=along_dimension,
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

    _reference_data = reference_sample.slice_along_dimension(along_dimension)
    tau_hypothesis_weights = predict_sample_ndf_hypothesis_weights(trained_model=trained_tau_model, predicted_distribution_corrected_size=experiment_sample.corrected_n_samples, reference_ndf_estimation=reference_sample)
    predicted_tau_ndf = plt.hist(_reference_data, weights=tau_hypothesis_weights, bins=bins, **prediction_hist_kwargs)
    ax.scatter(bin_centers, predicted_tau_ndf[0], label=tau_prediction_legend, color=tau_prediction_color, **prediction_scatter_kwargs)

    if trained_delta_model is not None:
        delta_hypothesis_weights = predict_sample_ndf_hypothesis_weights(trained_model=trained_delta_model, predicted_distribution_corrected_size=experiment_sample.corrected_n_samples, reference_ndf_estimation=reference_sample)
        predicted_delta_ndf = plt.hist(_reference_data, weights=delta_hypothesis_weights, bins=bins, **prediction_hist_kwargs)
        ax.scatter(bin_centers, predicted_delta_ndf[0], label=delta_prediction_legend, color=delta_prediction_color, **prediction_scatter_kwargs)

    ax.set_title(title, fontsize=30, pad=20)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig
