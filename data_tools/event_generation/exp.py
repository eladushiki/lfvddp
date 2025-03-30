from logging import info
from data_tools.dataset_config import DatasetConfig
import numpy as np

from dataclasses import dataclass
from typing import Callable


@dataclass
class ExpConfig(DatasetConfig):
    @property
    def dataset__analytic_background_function(self) -> Callable:
        return exp
 
    dataset_exp__is_poisson_fluctuations: int
    dataset_exp__signal_is_gaussian: bool
    dataset_exp__gaussian_signal_sigma: float


def exp(config: ExpConfig):
    '''
    Returns exponential samples A, B and Sig suitable for fitting.

    Parameters
    ----------
    config: An instance of ExpConfig containinhg all the parameters

    Returns
    -------
    A : numpy array
    B : numpy array
    Sig : numpy array
    '''
    N_Bkg_Pois  = np.random.poisson(lam=config.dataset__number_of_background_events*np.exp(config.dataset__nuisances_norm_reference_sigmas), size=1)[0] if config.dataset_exp__is_poisson_fluctuations else config.dataset__number_of_background_events
    N_Ref_Pois  = np.random.poisson(lam=config.dataset__number_of_reference_events*np.exp(config.dataset__nuisances_norm_reference_sigmas), size=1)[0] if config.dataset_exp__is_poisson_fluctuations else config.dataset__number_of_reference_events
    N_Sig_Pois = np.random.poisson(lam=config.dataset__number_of_signal_events*np.exp(config.dataset__nuisances_norm_reference_sigmas), size=1)[0] if config.dataset_exp__is_poisson_fluctuations else config.dataset__number_of_signal_events
    info(f"Drawn background samples: {N_Bkg_Pois}, reference sampels: {N_Ref_Pois} and signal samples: {N_Sig_Pois}")

    Bkg = np.random.exponential(scale=np.exp(config.dataset__nuisances_shape_reference_sigmas), size=(N_Bkg_Pois, 1))
    Ref = np.random.exponential(scale=1., size=(N_Ref_Pois, 1))
    if config.dataset_exp__signal_is_gaussian:
        Sig = np.random.normal(loc=config.dataset__signal_location, scale=config.dataset_exp__gaussian_signal_sigma, size=(N_Sig_Pois,1))*np.exp(config.dataset__nuisances_shape_reference_sigmas)
    else:
        def Sig_dist(x):
            dist = x**2*np.exp(-x)
            return dist/np.sum(dist)
        Sig = np.random.choice(np.linspace(0,100,100000),size=(N_Sig_Pois,1),replace=True,p=Sig_dist(np.linspace(0,100,100000)))*np.exp(config.dataset__nuisances_shape_reference_sigmas)
    
    info(f"Generated datasets;" \
          f" Ref of shape {Ref.shape}," \
          f" Bkg {Bkg.shape}" \
          f" and Sig {Sig.shape}")
    
    return Ref, Bkg, Sig
