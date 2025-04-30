from typing import Optional
from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetConfig
from frame.aggregate import ResultAggregator
from neural_networks.NPLM_adapters import predict_sample_ndf_hypothesis_weights
import numpy as np
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager, patches
from plot.plotting_config import PlottingConfig
from plot.carpenter import Carpenter
from scipy.stats import chi2,norm
from tensorflow.keras.models import Model  # type: ignore
from matplotlib.animation import FuncAnimation
import scipy.special as spc
from IPython.display import HTML

from frame.context.execution_context import ExecutionContext
from plot.plot_utils import HandlerCircle, HandlerRect, draw_sample_over_background_1D_histograms, em_results, exp_results, create_1D_containing_bins, get_z_score, results, scientific_number
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
    config = context.config

    # Training results aggregation
    agg = ResultAggregator(context)
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

    agg = ResultAggregator(context)
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
    ax.set_xlabel('t', labelpad=20)
    ax.set_ylabel('PDF', labelpad=20)
    ax.set_ylim(0, 0.1)
    plt.yticks([0.03, 0.06, 0.09])
    plt.xticks()

    return fig


def plot_old_t_2distributions(t_values1, t_values2, ref_str, bkg_str, df, epoch = 500000,xmin=0, xmax=300, ymin=0, ymax=0.1, nbins=10, label='', title='', save=False, save_path='', file_name=''):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution. 
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    
    t:  (numpy array shape (None,))
    df: (int) chi2 degrees of freedom
    '''
    raise NotImplementedError('This function is not implemented yet.')
    # t_dict = results_file.get_t_history_dict()
    # max_epoch = max(t_dict.keys())
    # t = t_dict[epoch] if epoch in t_dict.keys() else t_dict[max_epoch]
    t1 = t_values1
    t2 = t_values2
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    #set ax size
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height*0.5])
    # plot distribution histogram
    bins      = np.linspace(xmin, xmax, nbins+1)
    Z_obs1     = norm.ppf(chi2.cdf(np.median(t1), df))
    Z_obs2     = norm.ppf(chi2.cdf(np.median(t2), df))
    t_obs_err1 = 1.2533*np.std(t1)*1./np.sqrt(t1.shape[0])
    t_obs_err2 = 1.2533*np.std(t2)*1./np.sqrt(t2.shape[0])
    Z_obs_p1   = norm.ppf(chi2.cdf(np.median(t1)+t_obs_err1, df))
    Z_obs_p2   = norm.ppf(chi2.cdf(np.median(t2)+t_obs_err2, df))
    Z_obs_m1   = norm.ppf(chi2.cdf(np.median(t1)-t_obs_err1, df))
    Z_obs_m2   = norm.ppf(chi2.cdf(np.median(t2)-t_obs_err2, df))
    Ref_ratio = float(ref_str[0])/float(ref_str[1]) if len(ref_str)>1 else float(ref_str[0])
    Ref_events = int(219087*Ref_ratio)
    Bkg_ratio = float(bkg_str[0])/float(bkg_str[1]) if len(bkg_str)>1 else float(bkg_str[0])
    Bkg_events = int(219087*Bkg_ratio)
    # if label == "":
    #     events = r', $N_A^0=$'+f"{scientific_number(Ref_events)}"+r', $N_B^0=$'+f"{scientific_number(Bkg_events)}"
    #     label = 'exp'+events        
    # label  = 'sample: %s\nsize: %i \nmedian: %s, std: %s\n'%(label, t.shape[0], str(np.around(np.median(t), 2)),str(np.around(np.std(t), 2)))
    label1  = 'med: %s \nstd: %s'%(str(np.around(np.median(t1), 2)), str(np.around(np.std(t1), 2)))
    t2_median = np.median(t2[np.where(np.logical_not(np.isnan(t2)))])
    t2_std = np.std(t2[np.where(np.logical_not(np.isnan(t2)))])
    t1_num_of_nan = np.sum(np.isnan(t1))
    t2_num_of_nan = np.sum(np.isnan(t2))
    print('NumOfNans1: ', t1_num_of_nan, 'NumOfNans2: ', t2_num_of_nan)
    label2  = 'med: %s \nstd: %s'%(str(np.around(t2_median, 2)), str(np.around(t2_std, 2)))
    title = r'$N_A^0=$'+f"{scientific_number(Ref_events)}"+r',   $N_B^0=$'+f"{scientific_number(Bkg_events)}" if not title else title
    # label += 'Z = %s (+%s/-%s)'%(str(np.around(Z_obs, 2)), str(np.around(Z_obs_p-Z_obs, 2)), str(np.around(Z_obs-Z_obs_m, 2)))
    binswidth = (xmax-xmin)*1./nbins
    # if t1_num_of_nan > 0:
    #     t1[np.where(np.isnan(t1))[0]] = xmax - binswidth/2
    # if t2_num_of_nan > 0:
    #     t2[np.where(np.isnan(t2))[0]] = xmax - binswidth/2
    h1 = ax.hist(t1, weights=np.ones_like(t1)*1./(t1.shape[0]*binswidth), color='plum', ec='darkorchid',
                 bins=bins, label=label, alpha=0.8)
    h2 = ax.hist(t2, weights=np.ones_like(t2)*1./(t2.shape[0]*binswidth), color='lightcoral', ec='red',
                 bins=bins, label=label, alpha=0.5)
    err1 = np.sqrt(h1[0]/(t1.shape[0]*binswidth))
    err2 = np.sqrt(h2[0]/(t2.shape[0]*binswidth))
    x   = 0.5*(bins[1:]+bins[:-1])
    ax.errorbar(x, h1[0], yerr = err1, color='darkorchid', marker='o', ls='')
    ax.errorbar(x, h2[0], yerr = err2, color='red', marker='o', ls='')
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 1000)
    ax.plot(x, chi2.pdf(x, df),'grey', lw=5, alpha=0.8, label=f'$\chi^{2}_{{{df}}}$')
    font = font_manager.FontProperties(family='serif', size=24) 
    # plt.legend(prop=font,frameon=False)
    circ1 = patches.Circle((0,0), 1, facecolor='plum', edgecolor='darkorchid')
    circ2 = patches.Circle((0,0), 1, facecolor='lightcoral', edgecolor='red')
    rect = patches.Rectangle((0,0), 1, 1, color='grey', alpha=0.8)
    ax.legend((circ1, circ2, rect), (label1, label2, f'$\chi^{2}_{{{df}}}$'),
            handler_map={
               patches.Rectangle: HandlerRect(),
               patches.Rectangle: HandlerRect(),
               patches.Circle: HandlerCircle(),
            },
            prop=font,frameon=False)
    if (t1_num_of_nan > 0) or (t2_num_of_nan > 0):
        NaN_ratio = t1_num_of_nan/t1.shape[0] if t1_num_of_nan > 0 else t2_num_of_nan/t2.shape[0]
        rect2 = patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        legend = ax.legend((circ1, circ2, rect, rect2), (label1, label2, f'$\chi^{2}_{{{df}}}$',f'NaN: {NaN_ratio*100:.1f}%'),
            handler_map={
            patches.Rectangle: HandlerRect(),
            patches.Rectangle: HandlerRect(),
            patches.Circle: HandlerCircle(),
            },
            prop=font,frameon=False)
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel('t', fontsize=24, fontname="serif", labelpad=20)
    ax.set_ylabel('PDF', fontsize=24, fontname="serif", labelpad=20)
    ticks_list = [0.03,0.06,0.09]
    if ymax>=0.12: 
        ticks_list += [0.12]
    plt.yticks(ticks_list, fontsize=24, fontname="serif")
    plt.xticks(fontsize=24, fontname="serif")
    ax.set_title(title, fontsize=30, fontname="serif", pad=20)
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = '2distributions'
            else: 
                if (t1_num_of_nan > 0):
                    file_name += f'_{t1_num_of_nan}NaN'
                if (t2_num_of_nan > 0):
                    file_name += f'_{t2_num_of_nan}NaN'
                file_name += '_2distributions'
            plt.savefig(save_path+file_name+'.pdf')
    plt.show()
    plt.close(fig)


def plot_t_2distributions(results_file1:results, results_file2:results, df, epoch = 500000,xmin=0, xmax=300, nbins=10, label='', title='', save=False, save_path='', file_name=''):
    '''
    Plot the histogram of a test statistics sample (t) and the target chi2 distribution. 
    The median and the error on the median are calculated in order to calculate the median Z-score and its error.
    
    t:  (numpy array shape (None,))
    df: (int) chi2 degrees of freedom
    '''
    raise NotImplementedError('This function is not implemented yet.')
    # t1 = results_file1.get_t_history()[0][:,-1]
    # t2 = results_file2.get_t_history()[0][:,-1]

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
    fig  = plt.figure(figsize=(12, 9))
    fig.patch.set_facecolor('white')
    # plot distribution histogram
    for i in [1,2]:
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
        plt.errorbar(x, h[0], yerr = err, color=color[1], marker='o', ls='')
    # plot reference chi2
    x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 1000)
    plt.plot(x, chi2.pdf(x, df),'rebeccapurple', lw=5, alpha=0.8, label=f'$\chi^{2}_{{{df}}}$')
    font = font_manager.FontProperties(family='serif', size=14) 
    plt.legend(prop=font,frameon=False)
    plt.xlabel('t', fontsize=18, fontname="serif")
    plt.ylabel('PDF', fontsize=18, fontname="serif")
    plt.yticks(fontsize=16, fontname="serif")
    plt.xticks(fontsize=16, fontname="serif")
    plt.title(title, fontsize=18, fontname="serif")
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = '1distribution'
            else: file_name += '_1distribution'
            plt.savefig(save_path+file_name+'.pdf')
    plt.show()
    plt.close(fig)


def plot_t_multiple_distributions(results_files, df, epoch = 500000,xmin=0, xmax=300, ymax=0.1, nbins=10, bin_colors=[], edge_colors=[], alphas=[], labels=[], order=[], title='', save=False, save_path='', file_name=''):
    '''
    Plot histogram-distributions of the test scores for files in results_files, and the target chi2 distribution.

    results_files: (list) list of results files
    df:            (int) expected chi2 degrees of freedom. If df=0 or df=False, the chi2 distribution is not shown.
    epoch:         (int) epoch for which the test scores are plotted
    xmin:          (float) minimum value of the x-axis (in terms of t score).
    xmax:          (float) maximum value of the x-axis (in terms of t score).
    ymax:          (float) maximum value of the y-axis (probability).
    nbins:         (int) number of bins in the histogram.
    bin_colors:    (list) list of colors for the histograms. Empty list or 0-entries will use default colors, else list of colors for the bins.
    edge_colors:   (list) list of colors for the histograms. Empty list or 0-entries will use default colors, else list of colors for the edges.
    alphas:        (list) list of alpha values for the histograms. Empty list or empty-stirng-entries will use default values, else list of alpha values.
    labels:        (str) text for the legend of the histograms. Use 0 for default text.
    order:         (list) list of integers to set the order of the histograms' plotting in the plot. If empty, the results_files order is used, else list of integers.
    title:         (str) title of the plot.
    save:          (bool) if True, the plot is saved to the path specified in save_path, with the name specified in file_name.
    save_path:     (str) path to the directory where the plot will be saved.
    file_name:     (str) name of the file where the plot will be saved.
    '''
    raise NotImplementedError('This function is not implemented yet.')

    bin_colors = bin_colors if any(isinstance(l, str) for l in bin_colors) else [0]*len(results_files)
    edge_colors = edge_colors if any(isinstance(l, str) for l in edge_colors) else [0]*len(results_files)
    alphas = alphas if any(isinstance(l, (float,int)) for l in alphas) else ['']*len(results_files)
    labels = labels if any(isinstance(l, str) for l in labels) else [0]*len(results_files)
    t_dict = {}
    for i in range(len(results_files)):
        t_dict[i] = results_files[i].get_t_history_dict()
    max_epoch = min(max(t_dict[i].keys()) for i in range(len(results_files)))
    t = [t_dict[i][epoch] if epoch in t_dict[i].keys() else t_dict[i][max_epoch] for i in range(len(results_files))]
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    fig  = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.5, box.height*0.5])
    fig.patch.set_facecolor('white')
    for i in range(len(results_files)):
        color = bin_colors[i] if bin_colors[i]!=0 else 'plum' if results_files[i].NPLM=='False' else 'lightcoral'
        ec = edge_colors[i] if edge_colors[i]!=0 else 'darkorchid' if results_files[i].NPLM=='False' else 'red'
        alpha = alphas[i] if alphas[i]!='' else 0.8 if results_files[i].NPLM=='False' else 0.5
        bins      = np.linspace(xmin, xmax, nbins+1)
        Z_obs     = norm.ppf(chi2.cdf(np.median(t[i]), df))
        t_obs_err = 1.2533*np.std(t[i])*1./np.sqrt(t[i].shape[0])
        Z_obs_p   = norm.ppf(chi2.cdf(np.median(t[i])+t_obs_err, df))
        Z_obs_m   = norm.ppf(chi2.cdf(np.median(t[i])-t_obs_err, df))
        label  = 'med: %s \nstd: %s'%(str(np.around(np.median(t[i]), 2)), str(np.around(np.std(t[i]), 2))) if labels[i]==0 else labels[i]
        binswidth = (xmax-xmin)*1./nbins
        zorder = 2*i if order==[] else 2*order[i]
        h = ax.hist(t[i], weights=np.ones_like(t[i])*1./(t[i].shape[0]*binswidth), color=color, ec=ec,
                    bins=bins, label=label, alpha=alpha, zorder=zorder)
        err = np.sqrt(h[0]/(t[i].shape[0]*binswidth))
        x   = 0.5*(bins[1:]+bins[:-1])
        ax.errorbar(x, h[0], yerr = err, color=ec, marker='o', ls='', zorder=zorder+1)
    chi2_color = 'grey'
    if df:
        x  = np.linspace(chi2.ppf(0.0001, df), chi2.ppf(0.9999, df), 1000)
        ax.plot(x, chi2.pdf(x, df),chi2_color, lw=5, alpha=0.8, label=f'$\chi^{2}_{{{df}}}$')
    font = font_manager.FontProperties(family='serif', size=24)
    handles = []
    for i in range(len(results_files)):
        color = bin_colors[i] if bin_colors[i]!=0 else 'plum' if results_files[i].NPLM=='False' else 'lightcoral'
        ec = edge_colors[i] if edge_colors[i]!=0 else 'darkorchid' if results_files[i].NPLM=='False' else 'red'
        alpha = alphas[i] if alphas[i]!='' else 0.8 if results_files[i].NPLM=='False' else 0.5
        circ = patches.Circle((0,0), 1, facecolor=color, edgecolor=ec) if alpha else patches.Circle((0,0), 1, facecolor=ec, edgecolor='black')
        handles.append(circ)
    labels = ax.get_legend_handles_labels()[1]
    if df:
        rect = patches.Rectangle((0,0), 1, 1, color=chi2_color, alpha=0.8)
        handles.append(rect)
        labels.append(f'$\chi^{2}_{{{df}}}$')
    legend = ax.legend(handles, labels,
            handler_map={
            patches.Rectangle: HandlerRect(),
            patches.Circle: HandlerCircle(),
            },
            prop=font,frameon=False)
    ax.set_xlabel('t', fontsize=24, fontname="serif", labelpad=20)
    ax.set_ylabel('PDF', fontsize=24, fontname="serif", labelpad=20)
    ax.set_ylim(0,ymax)
    plt.yticks(np.arange(0,ymax+0.001,0.03)[1:], fontsize=24, fontname="serif")
    plt.xticks(fontsize=24, fontname="serif")
    ax.set_title(title, fontsize=30, fontname="serif", pad=20)
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            if file_name=='': file_name = 'multiple_distributions'
            else: file_name += '_multiple_distributions'
            plt.savefig(save_path+file_name+'.pdf')
    

def exp_performance_plot(Bkg_only_files, sig_type:int, title="", title_fs=24, labels_fs=21, ticks_fs=20, legend_fs=20, save=False, save_path='', saved_file_name='', errors = False):
    '''
    The function creates a plot of the measured significance as a function of the injected sqrt(q0) for the exponential distributions with a given signal type.

    Bkg_only_files: (list) list of Background files to be plotted.
                    For each background file, all available signal files are taken, and their significance is calculated and plotted (in a continuous line).
    sig_type:       (int) signal type (1,2,3) for which the significance is calculated.
                    1 - S1, 2 - S2, 3 - S3 (see paper for the mathematical definition of the signal types).
    title:          (str) title of the plot.
    title_fs:       (int) fontsize of the title.
    labels_fs:      (int) fontsize of the axis labels.
    ticks_fs:       (int) fontsize of the axis ticks.
    legend_fs:      (int) fontsize of the legend.
    save:           (bool) if True, the plot is saved to the path specified in save_path, with the name specified in saved_file_name.
    save_path:      (str) path to the directory where the plot will be saved.
    saved_file_name:(str) name of the file where the plot will be saved.
    errors:         (bool) if True, the plot includes errorbars.
    '''
    raise NotImplementedError('This function is not implemented yet.')

    fig, ax = plt.subplots(figsize=(9,6.75))
    fig.set_facecolor('white')
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    colors = plt.get_cmap('cool')
    for i,Bkg_only_file in enumerate(Bkg_only_files):
        if (Bkg_only_file.Bkg_sample == 'em') or (Bkg_only_file.Bkg_sample == 'em_Mcoll'):
            print('Used exp plot function for em sample.')
            return
        Bkg_only_file_name = Bkg_only_file.file
        Sig_file_names = Bkg_only_file.get_signals_files()[sig_type-1]
        #print( Sig_file_names)
        Sig_z_score = []
        approx_z_score = []
        Sig_q0 = []
        Z_score_p = []
        Z_score_m =[]
        for sig in Sig_file_names:
            Sig_file = exp_results(sig)
            if Sig_file.resample!="True":
                print(sig)
                z_score, Sig_t, Bkg_t = get_z_score(Sig_file,Bkg_only_file)
                if len(Sig_t)>0 and len(Bkg_t)>0:
                    Sig_z_score.append(z_score)
                    approx_z_score.append(norm.ppf(chi2.cdf(np.median(Sig_t), df=12)))
                    Sig_q0.append(Sig_file.get_sqrt_q0())
                    if errors:
                        z_score_p = np.sqrt(2)*spc.erfinv((len(Bkg_t[Bkg_t<=(np.median(Sig_t)+np.std(Sig_t))])/len(Bkg_t))*2-1)
                        z_score_m = np.sqrt(2)*spc.erfinv((len(Bkg_t[Bkg_t<=(np.median(Sig_t)-np.std(Sig_t))])/len(Bkg_t))*2-1)
                        Z_score_m.append(z_score_m)
                        Z_score_p.append(z_score_p)
        Sig_z_score = np.array(Sig_z_score)
        approx_z_score = np.array(approx_z_score)
        Sig_q0 = np.array(Sig_q0)
        sort = np.argsort(Sig_q0)
        Sig_z_score = Sig_z_score[sort]
        approx_z_score = approx_z_score[sort]
        
        Sig_q0 = Sig_q0[sort]
        label = r'$N_A^0=$'+f"{scientific_number(Bkg_only_file.Bkg_events)}"+r', $N_B^0=$'+f"{scientific_number(Bkg_only_file.Ref_events)}"
        ax.plot(Sig_q0, Sig_z_score, color=colors(1.0-i/len(Bkg_only_files)), label=label, linewidth=2)
        
        ax.plot(Sig_q0, approx_z_score, color=colors(1.0-i/len(Bkg_only_files)), linewidth=2, linestyle='--')
        if errors:
            Z_score_m = np.array(Z_score_m)
            Z_score_p = np.array(Z_score_p)
            Z_score_m = Z_score_m[sort]
            Z_score_p = Z_score_p[sort]
            ax.fill_between(Sig_q0, Z_score_m, Z_score_p,color=colors(1.0-i/len(Bkg_only_files)), label=label, linewidth=2,alpha = 0.1)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(0,13.5)
    ax.set_ylim(-2,4)
    ax.set_xlabel(r'injected $\sqrt{q_0}$', fontsize=labels_fs)
    ax.set_ylabel('measured significance', fontsize=labels_fs)
    ax.set_title(title, fontsize=title_fs)
    dashed_line = mpl.lines.Line2D([], [], color='black', linestyle='--', label=r"$\chi^2_{12}$")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(dashed_line)
    legend = ax.legend(handles=handles, labels=labels + [r"$\chi^2_{12}$"],loc='lower right', fontsize=legend_fs, fancybox=True, frameon=False)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0.0)
    ax.tick_params(labelsize=ticks_fs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(save_path+saved_file_name+'.pdf')
    plt.show()


def exp_multiple_performance_plot(Bkg_only_files, sig_type, title="", title_fs=24, labels_fs=21, ticks_fs=20, legend_fs=20, ignore_files=[], save=False, save_path='', saved_file_name='', errors = False):
    '''
    The function creates a plot of the measured significance as a function of the injected sqrt(q0) for the exponential distributions with a given signal type.
    Multiple plots are created, each for a different signal type.

    Bkg_only_files: (list) list of Background files to be plotted.
                    For each background file, all available signal files are taken, and their significance is calculated and plotted (in a continuous line).
    sig_type:       (list) list of signal types (1,2,3) for which the significance is calculated.
                    1 - S1, 2 - S2, 3 - S3 (see paper for the mathematical definition of the signal types).
    title:          (str) title of the plot.
    title_fs:       (int) fontsize of the title.
    labels_fs:      (int) fontsize of the axis labels.
    ticks_fs:       (int) fontsize of the axis ticks.
    legend_fs:      (int) fontsize of the legend.
    ignore_files:   (list) list of files to be ignored.
    save:           (bool) if True, the plot is saved to the path specified in save_path, with the name specified in saved_file_name.
    save_path:      (str) path to the directory where the plot will be saved.
    saved_file_name:(str) name of the file where the plot will be saved.
    errors:         (bool) if True, the plot includes (symmetrical!!!) errorbars.
    '''
    raise NotImplementedError('This function is not implemented yet.')

    fig = plt.figure(figsize=(8*(len(sig_type)+1), 14))
    fig.set_facecolor('white')
    gs = fig.add_gridspec(8, len(sig_type)+1, hspace=2)
    axs = ([fig.add_subplot(gs[1:5, i]) for i in range(len(sig_type))])
    # title_ax = fig.add_subplot(gs[0, :])
    # title_ax.set_title('Exp background', fontsize=title_fs)
    # title_ax.axis('off')
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    colors = plt.get_cmap('cool') #['violet', 'hotpink', 'mediumvioletred', 'mediumorchid', 'darkviolet']
    for j in range(len(sig_type)):
        for i,Bkg_only_file in enumerate(Bkg_only_files):
            # if i<3: colors = plt.get_cmap('Reds')
            # else: colors = plt.get_cmap('BuPu')
            if (Bkg_only_file.Bkg_sample == 'em') or (Bkg_only_file.Bkg_sample == 'em_Mcoll'):
                print('Used exp plot function for em sample.')
                return
            Bkg_only_file_name = Bkg_only_file.file
            Sig_file_names = Bkg_only_file.get_signals_files()[sig_type[j]-1]
            Sig_z_score = []
            approx_z_score = []
            Sig_q0 = []
            Z_score_p = []
            Z_score_m =[]
            for sig in Sig_file_names:
                Sig_file = exp_results(sig)
                if Sig_file.csv_file_name in ignore_files: continue
                if Sig_file.resample!="True":
                    print(sig)
                    z_score, Sig_t, Bkg_t = get_z_score(Sig_file,Bkg_only_file)
                    if len(Sig_t)>0 and len(Bkg_t)>0:
                        Sig_z_score.append(z_score)
                        approx_z_score.append(norm.ppf(chi2.cdf(np.median(Sig_t), df=12)))
                        Sig_q0.append(Sig_file.get_sqrt_q0())
                        if errors:
                            z_score_m = np.sqrt(2)*spc.erfinv((len(Bkg_t[Bkg_t<=(np.median(Sig_t)-np.std(Sig_t)/np.sqrt(len(Sig_t)))])/len(Bkg_t))*2-1)
                            if z_score_m==np.inf: z_score_m = norm.isf(chi2.sf(np.median(Sig_t)-np.std(Sig_t)/np.sqrt(len(Sig_t)), df=12))
                            z_score_p = np.sqrt(2)*spc.erfinv((len(Bkg_t[Bkg_t<=(np.median(Sig_t)+np.std(Sig_t)/np.sqrt(len(Sig_t)))])/len(Bkg_t))*2-1)
                            if z_score_p==np.inf: z_score_p = norm.isf(chi2.sf(np.median(Sig_t)+np.std(Sig_t)/np.sqrt(len(Sig_t)), df=12))
                            Z_score_m.append(z_score_m)
                            Z_score_p.append(z_score_p)
            Sig_z_score = np.array(Sig_z_score)
            approx_z_score = np.array(approx_z_score)
            Sig_q0 = np.array(Sig_q0)
            sort = np.argsort(Sig_q0)
            Sig_z_score = Sig_z_score[sort]
            approx_z_score = approx_z_score[sort]
            Sig_q0 = Sig_q0[sort]
            label = r'$N_A^0=$'+f"{scientific_number(Bkg_only_file.Bkg_events)}"+r', $N_B^0=$'+f"{scientific_number(Bkg_only_file.Ref_events)}"
            if (Bkg_only_file.NPLM == 'True') and ('NPLM' not in title[j]): label += ', NPLM'
            if not errors: axs[j].plot(Sig_q0, Sig_z_score, color=colors(1.0-i/len(Bkg_only_files)), label=label, linewidth=2, marker='o', markersize=4)
            if i<3: axs[j].plot(Sig_q0, approx_z_score, color=colors(0.75-i/len(Bkg_only_files)), linewidth=2, linestyle='--', zorder=-1)
            else: axs[j].plot(Sig_q0, approx_z_score, color=colors(0.75-(i-3)/len(Bkg_only_files)), linewidth=2, linestyle='--', zorder=-1)
            if errors:
                Z_score_m = np.array(Z_score_m)
                Z_score_p = np.array(Z_score_p)
                Z_score_m = Sig_z_score-Z_score_m[sort]
                # Z_score_p = Z_score_p[sort]-Sig_z_score   # for asymmetrical errorbars   
                Z_score_p = Z_score_m.copy()                # for symmetrical errorbars
                errors_array = np.concatenate((Z_score_m.reshape(1,-1),Z_score_p.reshape(1,-1)),axis=0)
                if i<3: axs[j].plot(Sig_q0, Sig_z_score, color=colors(0.75-i/len(Bkg_only_files)), label=label, linewidth=2, zorder=-1)
                else: axs[j].plot(Sig_q0, Sig_z_score, color=colors(0.75-(i-3)/len(Bkg_only_files)), label=label, linewidth=2, zorder=-1)
                axs[j].errorbar(Sig_q0, Sig_z_score, errors_array, linestyle='none', capsize=2, capthick=0.5, marker='o', mfc='black', ecolor='black', elinewidth=0.5, ms=2, zorder=3)
        axs[j].grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        axs[j].set_xlim(0,10)
        axs[j].set_ylim(-1,4)
        axs[j].set_xlabel(r'injected $\sqrt{q_0}$', fontsize=labels_fs)
        axs[j].set_ylabel('measured significance', fontsize=labels_fs)
        axs[j].set_title(title[j], fontsize=title_fs, pad=25)
        axs[j].tick_params(labelsize=ticks_fs)
        axs[j].xaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
        axs[j].yaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
        axs[j].set_xticks(ticks=[2,4,6,8,10])
    legend_ax = fig.add_subplot(gs[5,1]) 
    handles, labels = axs[0].get_legend_handles_labels()
    # if errors: handles = [h[0] for h in handles]
    dashed_line = mpl.lines.Line2D([], [], color='black', linestyle='--', label=r"$\chi^2_{12}$")
    handles.append(dashed_line)
    legend_ax.legend(handles, labels + [r"$\chi^2_{12}$"], loc='upper center', ncol=len(Bkg_only_files)//2+1, fontsize=legend_fs, fancybox=True, frameon=False, numpoints=1)
    # legend_ax.legend(handles, labels , loc='upper center', ncol=1, fontsize=legend_fs, fancybox=True, frameon=False, numpoints=1)
    legend_ax.axis('off')
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(save_path+saved_file_name+'.pdf')
    plt.show()


def em_performance_plot(Bkg_only_files, title="", title_fs=24, labels_fs=22, ticks_fs=20, legend_fs=20, save=False, save_path='', saved_file_name='', errors = False):
    '''
    The function creates a plot of the measured significance as a function of the injected sqrt(q0) for the em distribution.
    
    Bkg_only_files: (list) list of Background files to be plotted.
                    For each background file, all available signal files are taken, and their significance is calculated and plotted (in a continuous line).
    title:          (str) title of the plot.    
    title_fs:       (int) fontsize of the title.
    labels_fs:      (int) fontsize of the axis labels.
    ticks_fs:       (int) fontsize of the axis ticks.
    legend_fs:      (int) fontsize of the legend.
    save:           (bool) if True, the plot is saved to the path specified in save_path, with the name specified in saved_file_name.
    save_path:      (str) path to the directory where the plot will be saved.
    saved_file_name:(str) name of the file where the plot will be saved.
    errors:         (bool) if True, the plot includes (symmetrical!!!) errorbars.
    '''
    raise NotImplementedError('This function is not implemented yet.')
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(223)
    fig.set_facecolor('white')
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    colors = plt.get_cmap('spring')
    for i,Bkg_only_file in enumerate(Bkg_only_files):
        if Bkg_only_file.Bkg_sample == 'exp':
            print('Used em plot function for exp sample.')
            return
        Bkg_only_file_name = Bkg_only_file.file
        Sig_file_names = Bkg_only_file.get_signal_files()
        Sig_z_score = []
        approx_z_score = []
        Sig_q0 = []
        Z_score_p = []
        Z_score_m =[]
        for sig in Sig_file_names:
            Sig_file = em_results(sig)
            z_score, Sig_t, Bkg_t = get_z_score(Sig_file,Bkg_only_file)
            Sig_z_score.append(z_score)
            approx_z_score.append(norm.ppf(chi2.cdf(np.median(Sig_t), df=12)))
            Sig_q0.append(Sig_file.get_binned_sqrt_q0())
            if errors:
                z_score_m = np.sqrt(2)*spc.erfinv((len(Bkg_t[Bkg_t<=(np.median(Sig_t)-np.std(Sig_t)/np.sqrt(len(Sig_t)))])/len(Bkg_t))*2-1)    #dividing by sqrt(N_Sig)
                if z_score_m==np.inf: z_score_m = norm.isf(chi2.sf(np.median(Sig_t)-np.std(Sig_t)/np.sqrt(len(Sig_t)), df=12))
                z_score_p = np.sqrt(2)*spc.erfinv((len(Bkg_t[Bkg_t<=(np.median(Sig_t)+np.std(Sig_t)/np.sqrt(len(Sig_t)))])/len(Bkg_t))*2-1)    #dividing by sqrt(N_Sig)
                if z_score_p==np.inf: z_score_p = norm.isf(chi2.sf(np.median(Sig_t)+np.std(Sig_t)/np.sqrt(len(Sig_t)), df=12))
                Z_score_m.append(z_score_m)
                Z_score_p.append(z_score_p)
        Sig_z_score = np.array(Sig_z_score)
        approx_z_score = np.array(approx_z_score)
        Sig_q0 = np.array(Sig_q0)
        sort = np.argsort(Sig_q0)
        Sig_z_score = Sig_z_score[sort]
        approx_z_score = approx_z_score[sort]
        Sig_q0 = Sig_q0[sort]
        lumi = 20*Bkg_only_file.Bkg_ratio
        label = f'$\\mathcal{{L}}={int(lumi)}{{fb}}^{{-1}}$' if lumi.is_integer() else f'$\\mathcal{{L}}={lumi:.1f}{{fb}}^{{-1}}$'
        if errors:
            Z_score_m = np.array(Z_score_m)
            Z_score_p = np.array(Z_score_p)
            Z_score_m = Sig_z_score-Z_score_m[sort]
            # Z_score_p = Z_score_p[sort]-Sig_z_score   # for asymmetrical errorbars   
            Z_score_p = Z_score_m.copy()                # for symmetrical errorbars
            errors_array = np.concatenate((Z_score_m.reshape(1,-1),Z_score_p.reshape(1,-1)),axis=0)
            ax.plot(Sig_q0, Sig_z_score, color=colors(4/5*i/len(Bkg_only_files)), label=label, linewidth=2, zorder=-1)
            ax.errorbar(Sig_q0, Sig_z_score, errors_array, linestyle='none', capsize=2, capthick=0.5, marker='o', mfc='black', ecolor='black', elinewidth=0.5, ms=2, zorder=3)
        else:
            ax.plot(Sig_q0, Sig_z_score, color=colors(4/5*i/len(Bkg_only_files)), label=label, linewidth=2, marker='o', ms=4)
        ax.plot(Sig_q0, approx_z_score, color=colors(4/5*i/len(Bkg_only_files)), linewidth=2, linestyle='--', zorder=-1)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(0,10)
    ax.set_ylim(-1,4)
    ax.set_xlabel(r'injected $\sqrt{q_0}$', fontsize=labels_fs)
    ax.set_ylabel('measured significance', fontsize=labels_fs)
    ax.set_title(title, fontsize=title_fs, pad=25)
    handles, labels = ax.get_legend_handles_labels()
    # if errors: handles = [h[0] for h in handles]
    dashed_line = mpl.lines.Line2D([], [], color='black', linestyle='--', label=r"$\chi^2_{12}$")
    handles.append(dashed_line)
    legend = ax.legend(handles=handles, labels=labels + [r"$\chi^2_{12}$"], loc='lower right', fontsize=legend_fs, fancybox=True, frameon=False, numpoints=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0.0)
    ax.tick_params(labelsize=ticks_fs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(save_path+saved_file_name+'.pdf')
    plt.show()


def em_performance_plot_BR(Bkg_only_files, title="", title_fs=24, labels_fs=22, ticks_fs=20, legend_fs=20, save=False, save_path='', saved_file_name='', errors = False):
    '''
    The function creates a plot of the measured significance as a function of the branching ratio for the em distribution.
    See em_performance_plot's docstring for more details. 
    '''
    raise NotImplementedError('This function is not implemented yet.')

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(223)
    fig.set_facecolor('white')
    plt.rcParams["font.family"] = "serif"
    plt.style.use('classic')
    colors = plt.get_cmap('spring')
    BR_list = []
    Z_list = []
    for i,Bkg_only_file in enumerate(Bkg_only_files):
        if Bkg_only_file.Bkg_sample == 'exp':
            print('Used em plot function for exp sample.')
            return
        Bkg_only_file_name = Bkg_only_file.file
        Sig_file_names = Bkg_only_file.get_signal_files()
        Sig_z_score = []
        approx_z_score = []
        BR = []
        Z_score_p = []
        Z_score_m =[]
        lumi = 20*Bkg_only_file.Bkg_ratio
        for sig in Sig_file_names:
            Sig_file = em_results(sig)
            z_score, Sig_t, Bkg_t = get_z_score(Sig_file,Bkg_only_file)
            Sig_z_score.append(z_score)
            approx_z_score.append(norm.ppf(chi2.cdf(np.median(Sig_t), df=12)))
            BR.append((20*1/456)*Sig_file.Sig_events/lumi)
            if errors:
                z_score_m = np.sqrt(2)*spc.erfinv((len(Bkg_t[Bkg_t<=(np.median(Sig_t)-np.std(Sig_t)/np.sqrt(len(Sig_t)))])/len(Bkg_t))*2-1)    #dividing by sqrt(N_Sig)
                if z_score_m==np.inf: z_score_m = norm.isf(chi2.sf(np.median(Sig_t)-np.std(Sig_t)/np.sqrt(len(Sig_t)), df=12))
                z_score_p = np.sqrt(2)*spc.erfinv((len(Bkg_t[Bkg_t<=(np.median(Sig_t)+np.std(Sig_t)/np.sqrt(len(Sig_t)))])/len(Bkg_t))*2-1)    #dividing by sqrt(N_Sig)
                if z_score_p==np.inf: z_score_p = norm.isf(chi2.sf(np.median(Sig_t)+np.std(Sig_t)/np.sqrt(len(Sig_t)), df=12))
                Z_score_m.append(z_score_m)
                Z_score_p.append(z_score_p)
        Sig_z_score = np.array(Sig_z_score)
        approx_z_score = np.array(approx_z_score)
        BR = np.array(BR)
        sort = np.argsort(BR)
        Sig_z_score = Sig_z_score[sort]
        approx_z_score = approx_z_score[sort]
        BR = BR[sort]
        label = f'$\\mathcal{{L}}={int(lumi)}{{fb}}^{{-1}}$' if lumi.is_integer() else f'$\\mathcal{{L}}={lumi:.1f}{{fb}}^{{-1}}$'
        if errors:
            Z_score_m = np.array(Z_score_m)
            Z_score_p = np.array(Z_score_p)
            Z_score_m = Sig_z_score-Z_score_m[sort]
            # Z_score_p = Z_score_p[sort]-Sig_z_score   # for asymmetrical errorbars   
            Z_score_p = Z_score_m.copy()                # for symmetrical errorbars
            errors_array = np.concatenate((Z_score_m.reshape(1,-1),Z_score_p.reshape(1,-1)),axis=0)
            ax.plot(BR, Sig_z_score, color=colors(4/5*i/len(Bkg_only_files)), label=label, linewidth=2, zorder=-1)
            ax.errorbar(BR, Sig_z_score, errors_array, linestyle='none', capsize=2, capthick=0.5, marker='o', mfc='black', ecolor='black', elinewidth=0.5, ms=2, zorder=3)
        else:
            ax.plot(BR, Sig_z_score, color=colors(4/5*i/len(Bkg_only_files)), label=label, linewidth=2, marker='o', ms=4)
        ax.plot(BR, approx_z_score, color=colors(4/5*i/len(Bkg_only_files)), linewidth=2, linestyle='--', zorder=-1)
        BR_list.append(BR)
        Z_list.append(Sig_z_score)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_ylim(-1,4)
    ax.set_xlabel(r'BR %', fontsize=labels_fs)
    ax.set_ylabel('measured significance', fontsize=labels_fs)
    ax.set_title(title, fontsize=title_fs, pad=25)
    handles, labels = ax.get_legend_handles_labels()
    # if errors: handles = [h[0] for h in handles]
    dashed_line = mpl.lines.Line2D([], [], color='black', linestyle='--', label=r"$\chi^2_{12}$")
    handles.append(dashed_line)
    legend = ax.legend(handles=handles, labels=labels + [r"$\chi^2_{12}$"], loc='lower right', fontsize=legend_fs, fancybox=True, frameon=False, numpoints=1)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_linewidth(0.0)
    ax.tick_params(labelsize=ticks_fs)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True,prune='lower'))
    if save:
        if save_path=='': print('argument save_path is not defined. The figure will not be saved.')
        else:
            plt.savefig(save_path+saved_file_name+'.pdf')
    plt.show()
    return BR_list, Z_list


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


def plot_sample_over_background(
        context: ExecutionContext,
        sample: DataSet,
        background_sample: DataSet,
):
    """
    Generate a histogram of the sample over the background.
    """
    c = Carpenter(context)
    fig = c.figure()
    bins, _ = create_1D_containing_bins(
        context,
        [sample, background_sample]
    )

    A_ax = fig.add_subplot(1, 1, index=1)
    draw_sample_over_background_1D_histograms(
        ax=A_ax,
        sample=sample,
        background=background_sample,
        bins=bins,
        title="Sample over background",
    )

    return fig


def plot_1D_sliced_samples_over_background(
        context: ExecutionContext,
        first_sample: DataSet,
        second_sample: DataSet,
        background_sample: DataSet,
):
    """
    Generate two plots, both featuring historams of either sample over the background.
    Both are reconstructed to compensate for detector efficiency losses.
    """
    c = Carpenter(context)
    fig = c.figure()
    bins, _ = create_1D_containing_bins(
        context,
        [first_sample, second_sample, background_sample],
    )

    A_ax = fig.add_subplot(1, 2, 1)
    draw_sample_over_background_1D_histograms(
        ax=A_ax,
        sample=first_sample,
        background=background_sample,
        bins=bins,
        title="First sample over background",
    )
    B_ax = fig.add_subplot(1, 2, 2)
    draw_sample_over_background_1D_histograms(
        ax=B_ax,
        sample=second_sample,
        background=background_sample,
        bins=bins,
        title="Second sample over background",
    )

    return fig


def plot_1D_sliced_prediction_process(
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

    bins, bin_centers = create_1D_containing_bins(context, [experiment_sample, reference_sample])

    draw_sample_over_background_1D_histograms(
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
