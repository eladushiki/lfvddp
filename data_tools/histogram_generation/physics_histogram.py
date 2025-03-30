from abc import ABC
import numpy as np
from fractions import Fraction
from train.train_config import TrainConfig


from dataclasses import dataclass
from math import floor
from typing import Callable


@dataclass
class PhysicsConfig(TrainConfig, ABC):
    @classmethod
    def HISTOGRAM_NAME(cls) -> str:
        return "physics"

    @property
    def train__analytic_background_function(self) -> Callable:
        return physics
    @property
    def train__number_of_reference_events(self) -> int:
        return floor(self.train__batch_train_fraction)
    @property
    def train__number_of_background_events(self) -> int:
        return floor(self.train__batch_test_fraction)

    train_physics__n_poisson_fluctuations: int
    train_physics__data_usage_fraction: float  # As a fraction


def normalize(dataset, normalization=1e5):
    '''
    Normalizes the dataset according to the normalization.

    Parameters
    ----------
    normalization : float, str
        if float - the normalization factor. if str ('min-max' or 'standard') - the type of the normalization (incomplete for now).
    '''
    if normalization == 'min-max':
        dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    elif normalization == 'standard':
        dataset = (dataset - np.mean(dataset)) / np.std(dataset)
    elif isinstance(normalization,float) or isinstance(normalization,int):
        dataset = dataset/normalization
    else:
        raise ValueError("Invalid normalization, see docstring for options")
    return dataset


def physics(config: TrainConfig):
    '''
    Turns samples of the selected physical parameters into numpy arrays of samples A, B and Sig suitable for fitting.

    Parameters
    ----------
    config: An instance of PhysicsConfig containinhg all the parameters

    Returns
    -------
    A: numpy array
    B: numpy array
    Sig: numpy array
    '''
    if not isinstance(config, PhysicsConfig):
        raise TypeError(f"Expected PhysicsConfig, got {config.__class__.__name__}")

    # todo: move these to the config class
    channel = 'em'
    signal_types = ["ggH_taue","vbfH_taue"]
    used_physical_variables = ['Mcoll']
    binned = False
    resolution = 0.1

    background = {}
    signal = {}
    vars = used_physical_variables
    # background[f"{channel}_background"] = np.concatenate((np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{var}_dist.npy") for var in vars),axis=1)
    # for s in signal_types:
    #     signal[f"{s}_{channel}_signal"] = np.concatenate((np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{s}_signal_{var}_dist.npy") for var in vars),axis=1)
    # total_Sig = np.concatenate(tuple([signal[f"{s}_{channel}_signal"] for s in sig_types]),axis=0)
    background[f"{channel}_background"] = np.concatenate(tuple([np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{var}_dist.npy") for var in vars]),axis=1)
    for s in signal_types:
        signal[f"{s}_{channel}_signal"] = np.concatenate(tuple([np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{s}_signal_{var}_dist.npy") for var in vars]),axis=1) if config.train__signal_number_of_events>0 else np.empty((0,background[f"{channel}_background"].shape[1]))
    total_Sig = np.concatenate(tuple([signal[f"{s}_{channel}_signal"] for s in signal_types]),axis=0)

    N_A_Pois  = np.random.poisson(lam=float(Fraction(config.train__number_of_reference_events))*background[f"{channel}_background"].shape[0]*config.train_physics__data_usage_fraction, size=1)[0] if config.train_physics__n_poisson_fluctuations else float(Fraction(config.train__number_of_reference_events))*background[f"{channel}_background"].shape[0]*config.train_physics__data_usage_fraction
    N_B_Pois  = np.random.poisson(lam=float(Fraction(config.train__number_of_background_events))*background[f"{channel}_background"].shape[0]*config.train_physics__data_usage_fraction, size=1)[0] if config.train_physics__n_poisson_fluctuations else float(Fraction(config.train__number_of_background_events))*background[f"{channel}_background"].shape[0]*config.train_physics__data_usage_fraction
    N_Sig_Pois = np.random.poisson(lam=config.train__signal_number_of_events, size=1)[0] if config.train_physics__n_poisson_fluctuations else config.train__signal_number_of_events
    print(N_A_Pois,N_B_Pois,N_Sig_Pois)

    # bootstrapping
    # A = np.random.choice(background[f"{channel}_{var}_background"].reshape(-1,),N_A_Pois,replace=True).reshape(-1,1)
    # B = np.random.choice(background[f"{channel}_{var}_background"].reshape(-1,),N_B_Pois,replace=True).reshape(-1,1)
    # Sig = np.random.choice(total_Sig,N_Sig_Pois,replace=True).reshape(-1,1)
    A_events = np.random.choice(np.arange(background[f"{channel}_background"].shape[0]),N_A_Pois,replace=True)
    A = background[f"{channel}_background"][A_events]
    B_events = np.random.choice(np.arange(background[f"{channel}_background"].shape[0]),N_B_Pois,replace=True)
    B = background[f"{channel}_background"][B_events]
    Sig_events = np.random.choice(np.arange(total_Sig.shape[0]),N_Sig_Pois,replace=True)
    Sig = total_Sig[Sig_events]

    if binned:
        A = np.floor(A/resolution)*resolution
        B = np.floor(B/resolution)*resolution
        Sig = np.floor(Sig/resolution)*resolution
    A = normalize(A, normalization)
    B = normalize(B, normalization)
    Sig = normalize(Sig, normalization)
    print(f'setting: {channel}, N_A = {float(Fraction(config.train__number_of_reference_events))*background["em_background"].shape[0]*config.train_physics__data_usage_fraction}, N_B = {float(Fraction(config.train__number_of_background_events))*background["em_background"].shape[0]*config.train_physics__data_usage_fraction}, N_Sig = {config.train__signal_number_of_events}, N_poiss = {config.train_physics__n_poisson_fluctuations}')
    print('A: ',A.shape,' B: ',B.shape, ' Sig: ',Sig.shape)

    return A,B,Sig