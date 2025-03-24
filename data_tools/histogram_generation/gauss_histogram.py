from abc import ABC
import numpy as np
from train.train_config import TrainConfig

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable


@dataclass
class GaussConfig(TrainConfig, ABC):
    @classmethod
    def HISTOGRAM_NAME(cls) -> str:
        return "gauss"

    @property
    def dataset__analytic_background_function(self) -> Callable:
        return gauss

    train_gauss__is_poisson_fluctuations: bool
    trian_gauss__signal_covariant_magnitude: float


def gauss(config: TrainConfig):
    '''
    Returns gaussian samples A, B and Sig suitable for fitting.

    Parameters
    ----------
    config: An instance of GaussConfig containinhg all the parameters

    Returns
    -------
    reference: numpy array
    background: numpy array
    signal: numpy array
    '''
    if not isinstance(config, GaussConfig):
        raise TypeError(f"Expected GaussConfig, got {config.__class__.__name__}")

    # todo: move these to the config class
    normalization_factor = 1
    dim = 1

    n_bkg_pois  = np.random.poisson(lam = config.dataset__number_of_background_events * np.exp(normalization_factor), size = 1)[0] if config.train_gauss__is_poisson_fluctuations else config.dataset__number_of_background_events
    n_ref_pois  = np.random.poisson(lam = config.dataset__number_of_reference_events * np.exp(normalization_factor), size = 1)[0] if config.train_gauss__is_poisson_fluctuations else config.dataset__number_of_reference_events
    n_Sig_Pois = np.random.poisson(lam = config.dataset__number_of_signal_events * np.exp(normalization_factor), size = 1)[0] if config.train_gauss__is_poisson_fluctuations else config.dataset__number_of_signal_events
    background = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.ones((dim,dim)), size=n_bkg_pois)
    reference  = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.ones((dim,dim)), size=n_ref_pois)
    signal = np.random.multivariate_normal(mean = config.dataset__signal_location * np.ones(dim), cov = config.trian_gauss__signal_covariant_magnitude * np.ones((dim, dim)), size = n_Sig_Pois)
    return reference, background, signal
