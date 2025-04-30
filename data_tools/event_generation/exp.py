from data_tools.data_utils import DataSet
from data_tools.dataset_config import GeneratedDatasetParameters
from data_tools.event_generation.background import background_exp
from data_tools.event_generation.signal import signal_gaussian, signal_nonlocal


def exp(
        dataset_params: GeneratedDatasetParameters,
        gaussian_signal_sigma: float,
    ) -> DataSet:
    '''
    Returns exponentially distributed samples of any given dimension.
    
    This actually needs to be a general function but is a proxy for
    I don't want to overhaul this code right now.
    '''
    # Background
    background = background_exp(
        number_of_dimensions=dataset_params._dataset__number_of_dimensions,
        number_of_background_events=dataset_params.dataset__number_of_background_events,
        shape_nuisance_value=dataset_params.dataset__induced_shape_nuisance_value,
    )

    # Signal
    if dataset_params.dataset__signal_data_generation_function == "gaussian":
        signal = signal_gaussian(dataset_params, gaussian_signal_sigma)
    elif dataset_params.dataset__signal_data_generation_function == "nonlocal":
        signal = signal_nonlocal((dataset_params))
    else:
        raise ValueError(f"Unknown signal generation function: {dataset_params.dataset__signal_data_generation_function}")
  
    return background + signal
