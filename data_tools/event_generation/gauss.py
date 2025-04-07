from data_tools.data_utils import DataSet
from data_tools.dataset_config import GeneratedDatasetParameters
import numpy as np


def gauss(
        config: GeneratedDatasetParameters,
        is_poisson_fluctuations: bool,
        signal_covariant_magnitude: float,
        normalization_factor: float,
    ) -> DataSet:
    '''
    Returns gaussian samples A, B and Sig suitable for fitting.

    Parameters
    ----------
    config: An instance of GeneratedDatasetParameters containing all the parameters

    Returns
    -------
    a numpy array
    '''
    if is_poisson_fluctuations:
        number_of_background_events = np.random.poisson(lam = config.dataset__number_of_background_events * np.exp(normalization_factor), size = 1)[0]
        number_of_signal_events = np.random.poisson(lam = config.dataset__number_of_signal_events * np.exp(normalization_factor), size = 1)[0]
    else:
        number_of_background_events = config.dataset__number_of_background_events
        number_of_signal_events = config.dataset__number_of_signal_events

    background = np.random.multivariate_normal(mean=np.zeros(1), cov=np.ones((1, 1)), size=number_of_background_events)
    signal = np.random.multivariate_normal(mean = config.dataset__signal_location * np.ones(dim), cov = signal_covariant_magnitude * np.ones((dim, dim)), size=number_of_signal_events)

    events = np.concatenate((background, signal), axis=0)
    return DataSet(events)
