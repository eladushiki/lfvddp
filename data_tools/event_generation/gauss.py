from data_tools.dataset_config import DatasetConfig
import numpy as np

from dataclasses import dataclass
from typing import Callable


@dataclass
class GaussConfig(DatasetConfig):
    @property
    def dataset__analytic_background_function(self) -> Callable:
        return gauss

    dataset_gauss__is_poisson_fluctuations: bool
    dataset_gauss__signal_covariant_magnitude: float


def gauss(config: GaussConfig):
    '''
    Returns gaussian samples A, B and Sig suitable for fitting.

    Parameters
    ----------
    config: An instance of DatasetConfig containing all the parameters

    Returns
    -------
    reference: numpy array
    background: numpy array
    signal: numpy array
    '''
    # todo: move these to the config class
    normalization_factor = 1
    dim = 1

    n_bkg_pois  = np.random.poisson(lam = config.dataset__number_of_background_events * np.exp(normalization_factor), size = 1)[0] if config.dataset_gauss__is_poisson_fluctuations else config.dataset__number_of_background_events
    n_ref_pois  = np.random.poisson(lam = config.dataset__number_of_reference_events * np.exp(normalization_factor), size = 1)[0] if config.dataset_gauss__is_poisson_fluctuations else config.dataset__number_of_reference_events
    n_Sig_Pois = np.random.poisson(lam = config.dataset__number_of_signal_events * np.exp(normalization_factor), size = 1)[0] if config.dataset_gauss__is_poisson_fluctuations else config.dataset__number_of_signal_events
    background = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.ones((dim,dim)), size=n_bkg_pois)
    reference  = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.ones((dim,dim)), size=n_ref_pois)
    signal = np.random.multivariate_normal(mean = config.dataset__signal_location * np.ones(dim), cov = config.dataset_gauss__signal_covariant_magnitude * np.ones((dim, dim)), size = n_Sig_Pois)
    return reference, background, signal
