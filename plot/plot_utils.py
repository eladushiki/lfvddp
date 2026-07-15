import re
from glob import glob
from os.path import exists
from pathlib import Path
from readline import read_history_file
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from matplotlib import gridspec, patches, ticker
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerPatch

from data_tools.data_utils import DataSet
from data_tools.detector.detector_config import DetectorConfig
from frame.context.execution_context import ExecutionContext
from frame.file_structure import (
    TRAINING_HISTORY_LOG_FILE_SUFFIX,
    TRAINING_OUTCOMES_DIR_NAME,
)
from frame.file_system.training_history import HistoryKeys
from plot.plotting_config import PlottingConfig
from train.train_config import TrainConfig


class HandlerRect(HandlerPatch):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        x = width//3
        y = 0
        w = 25
        h = 10

        # create
        p = patches.Rectangle(xy=(x, y), width=w, height=h)

        # update with data from oryginal object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]
    
    
class HandlerCircle(HandlerPatch):

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height,
                       fontsize, trans):

        r = 5
        x = r + width//2
        y = height//2

        # create 
        p = patches.Circle(xy=(x, y), radius=r)

        # update with data from oryginal object
        self.update_prop(p, orig_handle, legend)

        # move xy to legend
        p.set_transform(trans)

        return [p]
    

def t_hist_epoch(epochs_list,t_history,epoch_numbers):
    t_hist_dict ={}
    for epoch in epoch_numbers:
        t_hist_dict[epoch] = []
        tepoch = [t_history[i][np.where(epochs_list[i]== epoch)[0]] for i in range(len(epochs_list))]
        if len(tepoch)>0:
            t_hist_dict[epoch] = np.concatenate(tepoch).ravel()
    return t_hist_dict 


def utils__get_spanning_dataset(
        config: DetectorConfig,
) -> DataSet:
    bin_centers = []
    for obs in (observable_names := config.detector__detect_observable_names):
        bin_centers.append(config.observable_bins(obs)[1])
    spanning_bin_centers = np.meshgrid(*bin_centers, indexing='ij')
    return DataSet(
        data=np.column_stack([b.ravel() for b in spanning_bin_centers]),
        observable_names=observable_names,
    )


class results:  # todo: deprecate
    N = 219087

    # members:
    _context: ExecutionContext
    _config: Union[PlottingConfig, TrainConfig]
    _history_files: List[Path]

    def __init__(self, containing_directory, context: ExecutionContext):  
        if not isinstance(config := context.config, TrainConfig):
            raise ValueError(f"Expected TrainConfig, got {type(config)}")
        if not isinstance(config, PlottingConfig):
            raise ValueError(f"Expected PlottingConfig, got {type(config)}")
        self._context = context
        self._config = config
        self._dir = containing_directory
        self.file = glob(containing_directory + "/**/*.csv", recursive=True)[0]
        self.csv_file_name = self.file
        self.tar_file_name = self.file.replace(".csv",".tar.gz") if self.file.endswith(".csv") else self.file
        self.Bkg_ratio = self._config.train__batch_train_fraction
        self.Bkg_events = int(results.N * self.Bkg_ratio)
        self.Ref_ratio = self._config.train__batch_test_fraction
        self.Ref_events = int(results.N * self.Ref_ratio)
        self.Sig_events = self._config.dataset__number_of_signal_events
        self.Bkg_sample = self._config.dataset__background_generation_function
        self.resolution = self._config.train__histogram_resolution
        self.WC = self._config.train__nn_weight_clipping

        self._history_files = [Path(s) for s in glob(f"{containing_directory}/**/*.{TRAINING_HISTORY_LOG_FILE_SUFFIX}", recursive=True)]
        self.Bkg_events = int(results.N * self._config.train__batch_train_fraction)
        self.Ref_events = int(results.N * self._config.train__batch_test_fraction)
        
        if hasattr(self._config, "train_physics__n_poisson_fluctuations"):
            self.N_poiss = self._config.train_physics__n_poisson_fluctuations
        elif hasattr(self._config, "train_gauss__n_poisson_fluctuations"):
            self.N_poiss = self._config.train_gauss__n_poisson_fluctuations
        elif hasattr(self._config, "train_exp__n_poisson_fluctuations"):
            self.N_poiss = self._config.train_exp__n_poisson_fluctuations
        else:
            self.N_poiss = "False"
        self.NPLM = "True"
        self.Sig_resonant = self._config.train__signal_is_gaussian
        self.Sig_loc = self._config.dataset__signal_location
        self.Sig_scale = self._config.train__signal_scale
        self.resample = self._config.dataset__resample_is_resample
        self.label_method = self._config.dataset__resample_label_method
        self.N_method = self._config.dataset__resample_method_type
        self.replacement = self._config.dataset__resample_is_replacement
        self.original_seed = self._context.random_seed
        self.tot_epochs = self._config.train__epochs

    def get_similar_files(self,epochs='all',patience_tau='all',patience_delta='all'):
        all_patience_str = self.file
        sub_epochs = '*' if epochs=='all' else f'{epochs}epochs_tau'
        sub_patience_tau = '*' if patience_tau=='all' else f'{patience_tau}patience_tau'
        sub_patience_delta = '*' if patience_delta=='all' else f'{patience_delta}patience_delta' 
        all_patience_str = re.sub(r'\d+epochs_delta','*',all_patience_str)
        all_patience_str = re.sub(r'\d+epochs_tau',sub_epochs,all_patience_str)
        all_patience_str = re.sub(r'\d+patience_delta',sub_patience_delta,all_patience_str)
        all_patience_str = re.sub(r'\d+patience_tau',sub_patience_tau,all_patience_str)
        #sample = "exp" if "exp" in file_name else re.search(r'em_?\S+', file_name)[0]
        #all_patience_str = self.sample+all_patience_str.split(self.sample)[1]#"exp"+all_patience_str.split('exp')[1] if "exp" in self.file else "em"+all_patience_str.split('em')[1]
        #all_patience_str = "exp"+all_patience_str.split('exp')[1] if "exp" in self.file else "em"+all_patience_str.split('em')[1]
        all_patience_str = all_patience_str.split('/')[-1]
        all_patience_str =re.sub(r'\*\*+', '*', all_patience_str)
        #all_patience_str  = re.sub(r'\d+signals',f"[^0-9]?{self.Sig_events}signals",all_patience_str)
        self.similar_search_name = all_patience_str
        
        
        #Sig_events = int(re.search(r'\d+signals', file_name)[0][:-len('signals')])
        files_all_patience_str = glob(TRAINING_OUTCOMES_DIR_NAME + all_patience_str)
        files = files_all_patience_str[:]
        for file_name in files:
            NPLM = "True" if ("TrueNPLM" in file_name  or ("delta" not in file_name and "Trueresample" not in file_name)) else "False"
            sig_events = int(re.search(r'\d+signals', file_name)[0][:-len('signals')])
            sample = "exp" if "exp" in file_name else "em_Mcoll" if "em_Mcoll" in file_name else "em"
            if self.NPLM !=NPLM or (self.Sig_events!=sig_events or self.Bkg_sample!=sample):
                files_all_patience_str.remove(file_name)
        self.similar_files = files_all_patience_str
        return files_all_patience_str   
    
    def __len__(self):
        return len(self._history_files)
    
    def read_final_t_csv(self):
        TAU_names = []
        TAUs = []
        delta_names = []
        deltas = []
        TAU_plus_delta =[]
        for file in self._history_files:
            if ".csv" not in file:
                file = file.replace("tar.gz","csv") if ".tar.gz" in file else file+".csv"
            csv_file = file
            file = csv_file.split(".csv")[0].split("/")[-1]
            if exists(csv_file):
                with open(csv_file,'r') as f:
                    lines = f.readlines()
                    TAU_names += [tau.split(',')[1] for tau in lines if (tau.count('TAU.') and tau.count(file))]
                    TAUs += [float(tau.split(',')[0]) for tau in lines if (tau.count('TAU.') and tau.count(file))]
                    # TAUs = np.array([float(tau.split(',')[0]) for tau in lines if tau.count('TAU.')])
                    delta_names += [delta.split(',')[1] for delta in lines if (delta.count('delta.') and delta.count(file))]
                    deltas += [float(delta.split(',')[0]) for delta in lines if (delta.count('delta.') and delta.count(file))]
                    # deltas = np.array([float(delta.split(',')[0]) for delta in lines if delta.count('delta.')])
                    
        if self.NPLM=="True":
            TAU_plus_delta =np.array(TAUs)
            delta_names =TAU_names.copy()
        else:            
            TAU_plus_delta = np.array([TAUs[TAU_names.index(delta_name.replace("delta.txt","TAU.txt"))]+deltas[delta_names.index(delta_name)] for delta_name in delta_names if TAU_names.count(delta_name.replace("delta.txt","TAU.txt"))>0])
            TAU_plus_delta_names = [TAU_names[TAU_names.index(delta_name.replace("delta.txt","TAU.txt"))]+' + '+delta_names[delta_names.index(delta_name)] for delta_name in delta_names if TAU_names.count(delta_name.replace("delta.txt","TAU.txt"))>0]
            delta_names = [delta_names[delta_names.index(delta_name)] for delta_name in delta_names if TAU_names.count(delta_name.replace("delta.txt","TAU.txt"))>0]

        return TAU_plus_delta, delta_names
             
    def get_t_history(self):
        t_final, txt_names = self.read_final_t_csv()
        txt_names = [(name.split("/")[-1]).replace("\n", "") for name in txt_names]
        history_files = f'{dir}extract_here/*{tar_file.replace(".tar.gz", "")}*_history*'
        files = glob(re.sub(r'\*\*+', '*', history_files))
        t_history = []
        epochs = []
        seeds = []
        if len(files) > 0:
            for filename in files:
                patience = self._config.train__number_of_epochs_for_checkpoint
                if self.NPLM=="False":
                    if (('_TAU_history' in filename) and (filename.replace('_TAU_history','_delta_history') in files)):
                        tau_or_delta_history = read_history_file(filename)
                        t_history.append(-2*(TAU_history[0::1]+delta_history[0::1]))
                        epochs.append(patience*np.array(range(len(TAU_history[0::1]))))
                elif self.NPLM=="True":
                    if '_TAU_history' in filename:
                        with h5py.File(filename, "r") as f1:
                            keys_list  = [(key) for key in list(f1.keys())]
                            TAU_history = f1.get(str(HistoryKeys.LOSS.value))
                            TAU_history = np.array(TAU_history)
                        t_history.append(-2*(TAU_history[0::1]))
                        epochs.append(patience*np.array(range(len(TAU_history[0::1]))))

        if len(t_final)>0:
            for i,t in enumerate(t_final):
                    name = txt_names[i]
                    tot_epochs = self._config.train__epochs
                    t_history.append(np.array([t]))
                    epochs.append(np.array([tot_epochs]))
        return t_history,epochs
    
    def get_t_history_dict(self):
        t_sig_hist, epochs_sig_list = self.get_t_history()
        epochs_list = np.unique(np.concatenate(epochs_sig_list).ravel())
        Sig_t = t_hist_epoch(epochs_sig_list, t_sig_hist,epochs_list)
        self.t_history = Sig_t.copy()
        return Sig_t
    
    def get_signal_files(self,N_sig='all',Sig_loc = 'all',Sig_scale = 'all',resonant="all"):
        filenames = []
        bkg_search_filename = self.similar_search_name.replace("tar.gz","csv")
        if self._config.train__nn_weight_clipping!=9:
            bkg_search_filename =(bkg_search_filename.split('clipping')[0]+'*'+bkg_search_filename.split('signals_')[1])
            sig_filename = re.sub(r'\*\*+', '*', bkg_search_filename)
        else:
            sig_filename = '*'+bkg_search_filename.split('signals_')[1]

        #print(sig_filename)
        sig_files = glob(results.dir+sig_filename)
        #print(sig_files)
        for file in sig_files:
            file = file.split('/')[-1]
            if "ch-" in file: continue
            params_file = results(file)
            flag = False
            if (params_file.Bkg_events==self.Bkg_events) and (params_file.Ref_events==self.Ref_events) and (params_file.N_poiss==self.N_poiss) and (params_file._config.train__histogram_resolution==self._config.train__histogram_resolution) and (params_file.NPLM==self.NPLM) and (params_file._config.train__histogram_analytic_pdf==self._config.train__histogram_analytic_pdf) and (params_file._config.train__nn_weight_clipping==self._config.train__nn_weight_clipping):
                flag = True
                if N_sig!="all":
                    flag = flag and (params_file._config.train__signal_number_of_events in N_sig)
                if params_file._config.train__signal_number_of_events!=0:
                    if Sig_loc!="all":
                        flag = flag and (params_file._config.dataset__signal_location in Sig_loc)
                    if Sig_scale!="all":
                        flag = flag and (params_file._config.train__signal_scale in Sig_scale)
                    if resonant!="all":
                        flag = flag and (params_file._config.train__signal_resonant in resonant)
                if flag and (file not in filenames):
                    filenames.append(file)
        return filenames


def utils__samples_over_background_histograms_sliced(
        ax: plt.Axes,
        samples: Union[DataSet, List[DataSet]],
        background: DataSet,
        bins: np.ndarray,
        along_observables: Union[List[str], str, None] = None,
        sample_legend: str = "sample",
        background_legend: str = "background",
):
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=background,
        along_observables=along_observables,
        label=background_legend,
    )

    if isinstance(samples, DataSet):
        samples = [samples]
    for ds in samples:
        utils__datset_histogram_sliced(
            ax=ax,
            bins=bins,
            dataset=ds,
            along_observables=along_observables,
            label=sample_legend,
        )


def utils__flatten_histogram_values(values: npt.ArrayLike) -> np.ndarray:
    return np.asarray(values).reshape(-1)


def utils__normalize_histogram_values(
    histogram_values: npt.ArrayLike,
    normalization: float,
) -> np.ndarray:
    """Normalize histogram values by their positive total event weight."""
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError("Histogram normalization must be a positive finite value.")
    return np.asarray(histogram_values) / normalization


def utils__datset_histogram_sliced(
        ax: plt.Axes,
        bins: np.ndarray,
        dataset: DataSet,
        alternative_weights: Optional[np.ndarray] = None,
        along_observables: Union[List, str, None] = None,
        normalize_by_corrected_n_samples: bool = False,
        **hist_kwargs,
):
    if along_observables is None:
        along_observables = dataset.observable_names[0]
    if isinstance(along_observables, str):
        along_observables = [along_observables]
    elif len(along_observables) > 2:
        raise ValueError("Can only plot 1D or 2D histograms, got more than 2 observables")

    weights = utils__flatten_histogram_values(
        dataset.histogram_weight_mask if alternative_weights is None else alternative_weights
    )
        
    if weights.shape[0] != dataset.n_samples:
        raise ValueError(
            f"Expected one histogram weight per sample, got weights shape {weights.shape} "
            f"for {dataset.n_samples} samples."
        )
    if normalize_by_corrected_n_samples:
        weights = utils__normalize_histogram_values(
            weights, dataset.corrected_n_samples
        )

    if len(along_observables) == 1:
        x = utils__flatten_histogram_values(
            dataset.slice_along_observable_names(along_observables[0])
        )
        ax.hist(
            x=x,
            bins=bins,
            weights=weights,
            log=True,
            **hist_kwargs,
        )
    else:
        xy = np.asarray(dataset.slice_along_observable_names(along_observables)).reshape(
            dataset.n_samples,
            len(along_observables),
        )
        x = xy[:, 0]
        y = xy[:, 1]
        hist2d_kwargs = dict(hist_kwargs)
        label = hist2d_kwargs.pop("label", None)
        histtype = hist2d_kwargs.pop("histtype", "stepfilled")
        log_scale = hist2d_kwargs.pop("log", False)
        alpha = hist2d_kwargs.pop("alpha", 0.8)
        linewidth = hist2d_kwargs.pop("lw", hist2d_kwargs.pop("linewidth", 0.8))
        color = hist2d_kwargs.pop("color", None)

        if getattr(ax, "name", "") == "3d":
            if color is None:
                color = ax._get_lines.get_next_color()

            counts, x_edges, y_edges = np.histogram2d(
                x=x,
                y=y,
                bins=bins,
                weights=weights,
            )
            x_pos, y_pos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
            dx, dy = np.meshgrid(np.diff(x_edges), np.diff(y_edges), indexing="ij")
            z_pos = np.zeros_like(counts)
            positive_mask = counts > 0

            bar_kwargs = dict(hist2d_kwargs)
            bar_kwargs.setdefault("shade", True)
            if histtype == "step":
                bar_kwargs.setdefault("color", (0.0, 0.0, 0.0, 0.0))
                bar_kwargs.setdefault("edgecolor", color)
                bar_kwargs.setdefault("alpha", 1.0)
                bar_kwargs.setdefault("linewidth", max(linewidth, 1.2))
            else:
                bar_kwargs.setdefault("color", color)
                bar_kwargs.setdefault("edgecolor", "black")
                bar_kwargs.setdefault("alpha", alpha)
                bar_kwargs.setdefault("linewidth", linewidth)

            x_bar = x_pos.ravel()[positive_mask.ravel()]
            y_bar = y_pos.ravel()[positive_mask.ravel()]
            z_bar = z_pos.ravel()[positive_mask.ravel()]
            dx_bar = dx.ravel()[positive_mask.ravel()]
            dy_bar = dy.ravel()[positive_mask.ravel()]
            dz_bar = counts.ravel()[positive_mask.ravel()]

            # Slightly offset overlaid histograms so mplot3d does not hide a tall bar
            # behind a shorter one that shares the same 3D footprint.
            plot_index = getattr(ax, "_dataset_histogram_sliced_3d_index", 0)
            offsets = [(-0.18, -0.18), (0.18, 0.18), (-0.18, 0.18), (0.18, -0.18)]
            x_offset_frac, y_offset_frac = offsets[plot_index % len(offsets)]
            shrink = 0.64
            x_bar = x_bar + ((1.0 - shrink) / 2.0 + x_offset_frac) * dx_bar
            y_bar = y_bar + ((1.0 - shrink) / 2.0 + y_offset_frac) * dy_bar
            dx_bar = dx_bar * shrink
            dy_bar = dy_bar * shrink
            ax._dataset_histogram_sliced_3d_index = plot_index + 1

            draw_order = np.argsort(dz_bar)

            ax.bar3d(
                x_bar[draw_order],
                y_bar[draw_order],
                z_bar[draw_order],
                dx_bar[draw_order],
                dy_bar[draw_order],
                dz_bar[draw_order],
                zsort="max",
                **bar_kwargs,
            )
            if label is not None:
                ax.scatter([], [], [], color=color, label=label)
            if log_scale and np.any(positive_mask):
                ax.set_zscale("log")
        else:
            if log_scale:
                hist2d_kwargs.setdefault("norm", LogNorm())
            ax.hist2d(
                x=x,
                y=y,
                bins=bins,
                weights=weights,
                **hist2d_kwargs,
            )


def utils__plot_datset_lfv_comparison(
    fig: plt.Figure,
    property_1: npt.NDArray,
    property_1_name: str,
    property_2: npt.NDArray,
    property_2_name: str,
    bin_edges: npt.NDArray,
    bin_centers: npt.NDArray,
    title: str,
    xlabel: str,
    ylabel: str,
):

    clean_x = [x_i[~np.isnan(x_i)] for x_i in [property_1, property_2]]

    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
    hist_ax = fig.add_subplot(gs[0])

    bin_heights_1, _, _ = hist_ax.hist(
        clean_x[0],
        bins=bin_edges,
        log=True,
        label=property_1_name,
        histtype='step',
        alpha=0.6,
    )
    bin_heights_2, _, _ = hist_ax.hist(
        clean_x[1],
        bins=bin_edges,
        log=True,
        label=property_2_name,
        histtype='step',
        alpha=0.6,
    )
    ratio = np.divide(bin_heights_1, bin_heights_2)
    
    # Remove x-axis label from top subplot
    hist_ax.tick_params(labelbottom=False)
    hist_ax.set_ylabel(ylabel)
    plt.legend()
    
    ratio_ax = fig.add_subplot(gs[1])
    ratio_ax.bar(
        bin_centers,
        ratio,
        label=f'{property_1} / {property_2}',
        width=0.8 * (bin_centers[1] - bin_centers[0]),  # thick bars
    )
    
    # Styling
    ratio_ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='y=1')
    ratio_ax.set_ylim(0.5, 1.5)
    ratio_ax.set_xlabel(xlabel)
    ratio_ax.set_ylabel("ratio")
    plt.suptitle(title)


def utils__contour_model_prediction(
        prediction_function: Callable[[DataSet], npt.NDArray],
        spanning_dataset: DataSet,
        along_observables: Union[List[str], str, None] = None,
        prediction_transform: Callable[[npt.NDArray], npt.NDArray] = np.exp,
):
    """
    Generate model prediction contour for 1D or 2D observables.

    For 1D: returns (x_range, contour_1d)
    For 2D: returns (x_range, y_range, contour_2d)
    """
    # Normalize input to list
    if along_observables is None:
        along_observables = [spanning_dataset.observable_names[0]]
    elif isinstance(along_observables, str):
        along_observables = [along_observables]

    sliced_dataset = spanning_dataset.filter_observable_names(along_observables)
    model_prediction = prediction_function(spanning_dataset)
    contour = prediction_transform(model_prediction)
    
    # Average over bins across projected dimensions
    unique_sliced_bin_centers, inverse_bin_indices = np.unique(
        sliced_dataset.events,
        axis=0,
        return_inverse=True,
    )

    contour_means = np.array([
        contour[inverse_bin_indices == bin_index].mean()
        for bin_index in range(len(unique_sliced_bin_centers))
    ])

    return unique_sliced_bin_centers, contour_means


def utils__add_subplot_sliced(
    fig: plt.Figure,
    subplot_shape: Tuple[int, int, int],
    number_of_dimensions: int,
) -> plt.Axes:
    """Create a regular 1D panel or a consistently oriented 2D-data 3D panel."""
    if number_of_dimensions == 1:
        return fig.add_subplot(*subplot_shape)

    panel = fig.add_subplot(*subplot_shape, projection="3d")
    panel.view_init(elev=28, azim=-58)
    panel.set_box_aspect((1.2, 1.2, 0.9))
    return panel


def utils__set_subplot_labels_sliced(
    ax: plt.Axes,
    bins: Union[np.ndarray, List[np.ndarray]],
    along_observables: List[str],
    output_label: str,
) -> None:
    """Apply shared observable limits and labels to a sliced plot panel."""
    if len(along_observables) == 1:
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel(along_observables[0])
        ax.set_ylabel(output_label)
        return

    ax.set_xlim(bins[0][0], bins[0][-1])
    ax.set_ylim(bins[1][0], bins[1][-1])
    ax.set_xlabel(along_observables[0])
    ax.set_ylabel(along_observables[1])
    ax.set_zlabel(output_label)


def utils__plot_region_histograms_sliced(
    ax: plt.Axes,
    sample_a: DataSet,
    sample_b: DataSet,
    background: DataSet,
    bins: Union[np.ndarray, List[np.ndarray]],
    along_observables: List[str],
    region_name: str,
    background_color: str,
    sample_a_color: str,
    sample_b_color: str,
    normalize_distributions: bool,
) -> None:
    """Draw a complete A/B/background distribution panel for one region."""
    distribution_specs = (
        (
            background,
            f"A-{region_name} + B-{region_name} (background)",
            background_color,
            "stepfilled",
            0.28,
            1.0,
        ),
        (sample_a, f"A-{region_name}", sample_a_color, "step", 1.0, 1.8),
        (sample_b, f"B-{region_name}", sample_b_color, "step", 1.0, 1.8),
    )
    for dataset, label, color, histtype, alpha, linewidth in distribution_specs:
        utils__datset_histogram_sliced(
            ax=ax,
            bins=bins,
            dataset=dataset,
            along_observables=along_observables,
            label=label,
            color=color,
            histtype=histtype,
            alpha=alpha,
            lw=linewidth,
            normalize_by_corrected_n_samples=normalize_distributions,
        )

    if len(along_observables) == 2:
        if normalize_distributions:
            minimum_output = min(
                1.0 / dataset.corrected_n_samples
                for dataset in (background, sample_a, sample_b)
                if dataset.corrected_n_samples > 0
            )
            output_limits = (
                minimum_output / 2.0,
                max(minimum_output, ax.get_zlim()[1]),
            )
        else:
            output_limits = (0.1, max(1.0, ax.get_zlim()[1]))
        ax.set_zlim(output_limits)
        ax.set_zscale("log")
        ax.zaxis.set_major_locator(ticker.LogLocator(base=10, numticks=4))
        ax.zaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))

    utils__set_subplot_labels_sliced(
        ax=ax,
        bins=bins,
        along_observables=along_observables,
        output_label=(
            "probability per bin"
            if normalize_distributions
            else "number of events per bin"
        ),
    )
    title_suffix = (
        "distributions (normalized)"
        if normalize_distributions
        else "number density functions"
    )
    ax.set_title(f"{region_name} {title_suffix}")


def utils__plot_weighted_histogram_predictions_sliced(
    ax: plt.Axes,
    reference_dataset: DataSet,
    predictions: List[Tuple[np.ndarray, str, str, str]],
    bins: Union[np.ndarray, List[np.ndarray]],
    bin_centers: Union[np.ndarray, List[np.ndarray]],
    along_observables: List[str],
    normalize_each_prediction: bool = False,
) -> None:
    """Overlay weighted reference-distribution predictions on a histogram panel."""
    number_of_dimensions = len(along_observables)
    reference_data = np.asarray(
        reference_dataset.slice_along_observable_names(along_observables)
    ).reshape(reference_dataset.n_samples, number_of_dimensions)

    if number_of_dimensions == 2:
        prediction_xx, prediction_yy = np.meshgrid(
            bin_centers[0], bin_centers[1], indexing="ij"
        )

    for weights, label, color, marker in predictions:
        flattened_weights = utils__flatten_histogram_values(weights)
        if number_of_dimensions == 1:
            predicted_counts, _ = np.histogram(
                reference_data[:, 0],
                bins=bins,
                weights=flattened_weights,
            )
        else:
            predicted_counts, _, _ = np.histogram2d(
                reference_data[:, 0],
                reference_data[:, 1],
                bins=bins,
                weights=flattened_weights,
            )

        if normalize_each_prediction:
            predicted_counts = utils__normalize_histogram_values(
                predicted_counts, np.sum(predicted_counts)
            )

        if number_of_dimensions == 1:
            ax.scatter(
                bin_centers,
                predicted_counts,
                label=label,
                color=color,
                marker=marker,
                s=28,
                edgecolor="black",
                linewidth=0.5,
            )
            continue

        positive_counts = predicted_counts.ravel() > 0
        ax.scatter(
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


def utils__synchronize_output_axis_limits(
    axes: List[plt.Axes],
    number_of_dimensions: int,
) -> None:
    """Share the event-count or model-output range across sliced panels."""
    output_axis = "y" if number_of_dimensions == 1 else "z"
    get_limits_name = f"get_{output_axis}lim"
    set_limits_name = f"set_{output_axis}lim"
    shared_limits = (
        min(getattr(ax, get_limits_name)()[0] for ax in axes),
        max(getattr(ax, get_limits_name)()[1] for ax in axes),
    )
    for ax in axes:
        getattr(ax, set_limits_name)(shared_limits)


def utils__model_prediction_values(
    prediction_function: Callable[[DataSet], npt.NDArray],
    dataset: DataSet,
) -> np.ndarray:
    """Evaluate and flatten one model prediction per event."""
    return utils__flatten_histogram_values(prediction_function(dataset))


def utils__remove_eta_from_prediction_values(
    prediction_values: np.ndarray,
    eta_values: np.ndarray,
    eta_sign: float,
) -> np.ndarray:
    """Remove the clamped 1±eta factor from a combined LFVDDP prediction."""
    eta_term = np.clip(1.0 + eta_sign * eta_values, a_min=1e-12, a_max=None)
    return prediction_values / eta_term


def utils__project_prediction_values_sliced(
    values: np.ndarray,
    spanning_dataset: DataSet,
    along_observables: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Project already evaluated model values onto selected observables."""
    return utils__contour_model_prediction(
        prediction_function=lambda _: values,
        spanning_dataset=spanning_dataset,
        along_observables=along_observables,
        prediction_transform=np.asarray,
    )


def utils__plot_model_predictions_sliced(
    ax: plt.Axes,
    predictions: List[Tuple[np.ndarray, np.ndarray, str, str, str]],
    bins: Union[np.ndarray, List[np.ndarray]],
    along_observables: List[str],
    prediction_limits: Tuple[float, float],
    title: str,
    draw_as_steps: bool = False,
    continuous_predictions: Optional[
        List[Tuple[np.ndarray, np.ndarray, str, str, str]]
    ] = None,
) -> None:
    """Draw a complete 1D-line or 2D-wireframe model-prediction panel."""
    first_coordinates = predictions[0][0]
    if len(along_observables) == 1:
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        for coordinates, contour, label, color, linestyle in predictions:
            plot_kwargs = {
                "label": label,
                "color": color,
                "linestyle": linestyle,
                "linewidth": 1.8,
            }
            if draw_as_steps:
                ax.stairs(
                    utils__flatten_histogram_values(contour),
                    bins,
                    **plot_kwargs,
                )
            else:
                ax.plot(
                    utils__flatten_histogram_values(coordinates),
                    utils__flatten_histogram_values(contour),
                    **plot_kwargs,
                )
        for coordinates, contour, _, color, linestyle in continuous_predictions or []:
            flattened_coordinates = utils__flatten_histogram_values(coordinates)
            ax.plot(
                flattened_coordinates,
                utils__flatten_histogram_values(contour),
                color=color,
                linestyle=linestyle,
                linewidth=1.0,
                alpha=0.9,
                marker="x",
                markersize=3.5,
                markeredgewidth=0.6,
                markevery=max(1, len(flattened_coordinates) // 40),
                label="_nolegend_",
                zorder=4,
            )
        if continuous_predictions:
            ax.plot(
                [],
                [],
                color="black",
                linewidth=1.0,
                marker="x",
                markersize=3.5,
                label="dense model evaluation",
            )
        if draw_as_steps:
            ax.set_xticks(bins)
            for bin_edge in np.asarray(bins)[1:-1]:
                ax.axvline(
                    bin_edge,
                    color="gray",
                    linestyle=":",
                    linewidth=0.7,
                    alpha=0.35,
                    zorder=0,
                )
        ax.set_ylim(prediction_limits)
    else:
        if draw_as_steps:
            x_values = np.asarray(bins[0])
            y_values = np.asarray(bins[1])
        else:
            x_values = np.unique(first_coordinates[:, 0])
            y_values = np.unique(first_coordinates[:, 1])
        prediction_xx, prediction_yy = np.meshgrid(
            x_values, y_values, indexing="ij"
        )
        ax.plot_surface(
            prediction_xx,
            prediction_yy,
            np.ones_like(prediction_xx),
            color="gray",
            linewidth=0,
            alpha=0.06,
            shade=False,
        )
        for coordinates, contour, label, color, linestyle in predictions:
            if draw_as_steps:
                x_values = np.repeat(np.asarray(bins[0]), 2)[1:-1]
                y_values = np.repeat(np.asarray(bins[1]), 2)[1:-1]
                contour_grid = np.asarray(contour).reshape(
                    len(bins[0]) - 1, len(bins[1]) - 1
                )
                contour_grid = np.repeat(
                    np.repeat(contour_grid, 2, axis=0), 2, axis=1
                )
            else:
                x_values = np.unique(coordinates[:, 0])
                y_values = np.unique(coordinates[:, 1])
                contour_grid = np.asarray(contour).reshape(
                    len(x_values), len(y_values)
                )
            prediction_xx, prediction_yy = np.meshgrid(
                x_values, y_values, indexing="ij"
            )
            ax.plot_wireframe(
                prediction_xx,
                prediction_yy,
                contour_grid,
                color=color,
                linestyle=linestyle,
                linewidth=0.8,
                alpha=0.9,
            )
            ax.plot([], [], [], label=label, color=color, linestyle=linestyle)
        for coordinates, contour, _, color, linestyle in continuous_predictions or []:
            x_values = np.unique(coordinates[:, 0])
            y_values = np.unique(coordinates[:, 1])
            continuous_grid = np.asarray(contour).reshape(
                len(x_values), len(y_values)
            )
            continuous_xx, continuous_yy = np.meshgrid(
                x_values, y_values, indexing="ij"
            )
            ax.plot_wireframe(
                continuous_xx,
                continuous_yy,
                continuous_grid,
                color=color,
                linestyle=linestyle,
                linewidth=0.65,
                alpha=0.8,
            )
        if draw_as_steps:
            ax.set_xticks(bins[0])
            ax.set_yticks(bins[1])
        ax.set_zlim(prediction_limits)

    utils__set_subplot_labels_sliced(
        ax=ax,
        bins=bins,
        along_observables=along_observables,
        output_label="model prediction",
    )
    ax.set_title(title)
    ax.legend(fontsize=7)
