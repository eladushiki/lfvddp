from data_tools.data_utils import DataSet
from data_tools.dataset_config import GeneratedDatasetParameters
import numpy as np


def exp(
        config: GeneratedDatasetParameters,
        is_poisson_fluctuations: int,
        is_signal_gaussian: bool,
        gaussian_signal_sigma: float,
    ) -> DataSet:
    '''
    Returns exponential samples A, B and Sig suitable for fitting.

    Parameters
    ----------
    config: An instance of GeneratedDatasetParameters containing all the parameters

    Returns
    -------
    a numpy array
    '''
    if is_poisson_fluctuations:
        number_of_background_events  = np.random.poisson(lam=config.dataset__number_of_background_events*np.exp(config.dataset__induced_nuisances_norm_reference_sigmas), size=1)[0]
        number_of_signal_events = np.random.poisson(lam=config.dataset__number_of_signal_events*np.exp(config.dataset__induced_nuisances_norm_reference_sigmas), size=1)[0]
    else:
        number_of_background_events = config.dataset__number_of_background_events
        number_of_signal_events = config.dataset__number_of_signal_events
    
    background = np.random.exponential(scale=np.exp(config.dataset__induced_nuisances_shape_reference_sigmas), size=(number_of_background_events, 1))
    if is_signal_gaussian:
        signal = np.random.normal(loc=config.dataset__signal_location, scale=gaussian_signal_sigma, size=(number_of_signal_events,1))*np.exp(config.dataset__induced_nuisances_shape_reference_sigmas)
    else:
        def Sig_dist(x):
            dist = x**2*np.exp(-x)
            return dist/np.sum(dist)
        signal = np.random.choice(np.linspace(0,100,100000),size=(number_of_signal_events,1),replace=True,p=Sig_dist(np.linspace(0,100,100000)))*np.exp(config.dataset__induced_nuisances_shape_reference_sigmas)
    
    events = np.concatenate((background, signal), axis=0)
    return DataSet(events)
