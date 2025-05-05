from typing import TypeVar, Union
from data_tools.data_utils import DataSet
from data_tools.dataset_config import GeneratedDatasetParameters
import numpy as np

# Namespace for signal events generating functions
# functions defined here are automatically recognized by the
# program and can be called from the config file.
# Have the function signatures as follows:
#
# def signal_function_name(
#         params: GeneratedDatasetParameters,
#         **kwargs
#     ) -> DataSet:
#     ...
#


def signal_gaussian(
        params: GeneratedDatasetParameters,
        gaussian_signal_sigma: float,
        **kwargs
) -> DataSet:
    return DataSet(np.random.normal(
        loc=params.dataset__signal_parameters["location"],
        scale=gaussian_signal_sigma,
        size=(params.dataset__number_of_signal_events, params._dataset__number_of_dimensions) * \
            round(np.exp(params.dataset__induced_shape_nuisance_value)),
    ))

FLOAT_OR_ARRAY = TypeVar('FLOAT_OR_ARRAY', bound=Union[float, np.ndarray])
def gaussian(
        x: FLOAT_OR_ARRAY,
        mean: float,
        sigma: float,
) -> FLOAT_OR_ARRAY:
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * sigma**2))


def signal_nonlocal(
        params: GeneratedDatasetParameters,
        **kwargs
) -> DataSet:
    rng = np.linspace(0, 100, 100000)

    return DataSet(np.random.choice(
        rng,
        size=(params.dataset__number_of_signal_events, params._dataset__number_of_dimensions),
        replace=True,
        p=normalized_nonlocal(rng) * np.exp(params.dataset__induced_shape_nuisance_value),
    ))


FLOAT_OR_ARRAY = TypeVar('FLOAT_OR_ARRAY', bound=Union[float, np.ndarray])
def normalized_nonlocal(x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
    dist = x**2 * np.exp(-x)
    normalization = 2  # Definite integral in [0, inf) is 2``
    return dist / normalization
