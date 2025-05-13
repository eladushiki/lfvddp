from glob import glob
from os.path import exists
from pathlib import Path
from readline import read_history_file
from typing import Callable, List, Optional, Union

from data_tools.data_utils import DataSet
from data_tools.profile_likelihood import calc_median_t_significance_relative_to_background
from frame.context.execution_context import ExecutionContext
from frame.file_structure import TRAINING_HISTORY_FILE_EXTENSION, TRIANING_OUTCOMES_DIR_NAME
from frame.file_system.training_history import HistoryKeys
import numpy as np
from matplotlib import patches, pyplot as plt
from plot.plotting_config import PlottingConfig
import scipy.special as spc
from matplotlib.legend_handler import HandlerPatch
import re

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

        self._history_files = [Path(s) for s in glob(f"{containing_directory}/**/*.{TRAINING_HISTORY_FILE_EXTENSION}", recursive=True)]
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
        all_patience_str =re.sub('\*\*+', '*', all_patience_str)
        #all_patience_str  = re.sub(r'\d+signals',f"[^0-9]?{self.Sig_events}signals",all_patience_str)
        self.similar_search_name = all_patience_str
        
        
        #Sig_events = int(re.search(r'\d+signals', file_name)[0][:-len('signals')])
        files_all_patience_str = glob(TRIANING_OUTCOMES_DIR_NAME + all_patience_str)
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
        files = glob(re.sub('\*\*+', '*', history_files))
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
            sig_filename = re.sub('\*\*+', '*', bkg_search_filename)
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


class em_results(results):
    # channel='em'
    # signal_samples=["ggH_taue","vbfH_taue"]#["ggH_taue","ggH_taumu","vbfH_taue","vbfH_taumu","Z_taue","Z_taumu"]
    # background = {}
    # signal = {}
    # background["em_background"] = np.load("/storage/agrp/yuvalzu/NPLM/em_MLL_dist.npy")
    # for s in signal_samples:
    #     signal[f"{s}_em_signal"] = np.load(f"/storage/agrp/yuvalzu/NPLM/em_{s}_signal_MLL_dist.npy")
    # Bkg = background["em_background"]
    
    def __init__(self, file_name):
        super().__init__(file_name)

    def collect_data(self):
        channel='em'
        signal_samples=["ggH_taue","vbfH_taue"]#["ggH_taue","ggH_taumu","vbfH_taue","vbfH_taumu","Z_taue","Z_taumu"]
        background = {}
        signal = {}
        background["em_background"] = np.load("/storage/agrp/yuvalzu/NPLM/em_MLL_dist.npy")
        for s in signal_samples:
            signal[f"{s}_em_signal"] = np.load(f"/storage/agrp/yuvalzu/NPLM/em_{s}_signal_MLL_dist.npy")
        Bkg = background["em_background"]
        return Bkg, signal, signal_samples

    def get_sqrt_q0(self,Data_bins=6):
        # Bkg = em_results.Bkg
        # signal_samples = em_results.signal_samples
        # signal = em_results.signal
        Bkg, signal, signal_samples = self.collect_data()
        Sig = np.concatenate(tuple([signal[f"{s}_em_signal"] for s in signal_samples]),axis=0).reshape(-1,)
        mu = self._config.train__signal_number_of_events/len(Sig)
        bkgFrac = self.Bkg_events/len(Bkg)
        histData = np.histogram(Sig,Data_bins)
        bins = histData[1]
        data = histData[0][histData[0]!=0]
        histBkg = np.histogram(Bkg,bins)
        bkg = histBkg[0][histData[0]!=0]
        sqrt_q0 = 2*(-self._config.train__signal_number_of_events+np.sum((mu*data+(bkg*bkgFrac))*np.log(mu*data/(bkg*bkgFrac)+1)))
        return np.sqrt(sqrt_q0)
    
    def get_binned_sqrt_q0(self,resolution=0.05):
        Bkg, signal, signal_samples = self.collect_data()
        Sig = np.concatenate(tuple([signal[f"{s}_em_signal"] for s in signal_samples]),axis=0).reshape(-1,)
        Bkg = np.floor((Bkg/1e5)/resolution)*resolution
        Sig = np.floor((Sig/1e5)/resolution)*resolution
        mu = self._config.train__signal_number_of_events/len(Sig)
        bkgFrac = self.Bkg_events/len(Bkg)
        decimals = len(str(resolution).split('.')[1])
        histData = np.histogram(Sig, bins=[min(Sig)+round(i*resolution,decimals) for i in range(int((max(Sig)-min(Sig)+round(2*resolution,decimals))/resolution))])
        bins = histData[1]
        data = histData[0][histData[0]!=0]
        histBkg = np.histogram(Bkg,bins)
        bkg = histBkg[0][histData[0]!=0]
        sqrt_q0 = 2*(-self._config.train__signal_number_of_events+np.sum((mu*data+(bkg*bkgFrac))*np.log(mu*data/(bkg*bkgFrac)+1)))
        return np.sqrt(sqrt_q0)
    
    def get_signals_files(self):
        filenames =self.get_signal_files(Sig_loc = [6.4],Sig_scale = [0.16],resonant=["True"])
        return filenames
        
    

def scientific_number(num, digits=2):
    if num==0:
        return "0"
    exp = int(np.floor(np.log10(np.abs(num))))
    coeff = round(np.abs(num)/(10**exp),digits)
    if num<0:
        coeff = -coeff
    if exp==0:
        sn_string = f"{coeff}"
    elif exp==-1:
        sn_string = f"0.{str(10*coeff).split('.')[0]}"
    else:
        sn_string = f"${coeff}\\times{{10}}^{{{exp}}}$"
    # print(sn_string)
    return sn_string


def get_z_score(
        results_file: Path,
        bkg_results_file: Path,
        epoch = 500000
    ):
    res = results(results_file)
    background_results = results(bkg_results_file)
    sig_results = res
    bkg_results = background_results
    sig_t = []
    bkg_t = []
    z_score = float("nan")
    if epoch == sig_results._config.train__epochs:
        sig_t = sig_results.read_final_t_csv()[0]
    if (len(sig_t)<1) or (epoch < sig_results._config.train__epochs):
        sig_t_dict = sig_results.get_t_history_dict()
        sig_t = sig_t_dict[epoch] if epoch in sig_t_dict.keys() else sig_t_dict[sig_results._config.train__epochs]
    if epoch > sig_results._config.train__epochs: 
        print(f"maximal number of signal epochs = {sig_results._config.train__epochs}")
        return z_score, sig_t, bkg_t

    if epoch == bkg_results._config.train__epochs:
        bkg_t = bkg_results.read_final_t_csv()[0]
    if (len(bkg_t)<1) or (epoch < bkg_results._config.train__epochs):
        bkg_t_dict = bkg_results.get_t_history_dict()
        bkg_t = bkg_t_dict[epoch] if epoch in bkg_t_dict.keys() else bkg_t_dict[sig_results._config.train__epochs]
    if epoch > bkg_results._config.train__epochs: 
        print(f"maximal number of bkg epochs = { background_results._config.train__epochs}")
        return z_score, sig_t, bkg_t

    # replace NaNs in bkg_t and sig_t with inf.
    bkg_t[np.isnan(bkg_t)] = np.inf
    sig_t[np.isnan(sig_t)] = np.inf
    z_score = calc_median_t_significance_relative_to_background(sig_t, bkg_t)
    return z_score, sig_t, bkg_t


def utils__create_slice_containing_bins(
        datasets: List[DataSet],
        nbins = 100,
        along_dimension: int = 0,
):
    # limits    
    xmin = 0
    xmax = np.max([np.max(dataset.slice_along_dimension(along_dimension)) for dataset in datasets])

    # bins
    bins = np.linspace(xmin, xmax, nbins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    return bins, bin_centers


def utils__sample_over_background_histograms_sliced(
        ax: plt.Axes,
        sample: DataSet,
        background: DataSet,
        bins: np.ndarray,
        along_dimension: int = 0,
        sample_legend: str = "sample",
        background_legend: str = "background",
):
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=background,
        along_dimension=along_dimension,
        label=background_legend,
    )
    utils__datset_histogram_sliced(
        ax=ax,
        bins=bins,
        dataset=sample,
        along_dimension=along_dimension,
        label=sample_legend,
    )


def utils__datset_histogram_sliced(
        ax: plt.Axes,
        bins: np.ndarray,
        dataset: DataSet,
        alternative_weights: Optional[np.ndarray] = None,
        along_dimension: int = 0,
        label: Optional[str] = None,
        histtype: str = "bar",
):
    ax.hist(
        x=dataset.slice_along_dimension(along_dimension),
        bins=bins,
        weights=dataset.histogram_weight_mask if alternative_weights is None else alternative_weights,
        log=True,
        histtype=histtype,
        label=label,
    )
