from typing import TypeVar, Union
from data_tools.data_utils import DataSet
import numpy as np

# Namespace for background generating functions
# functions defined here are automatically recognized by the
# program and can be called from the config file.
# Have the function signatures as follows:
#
# def background_function_name(
#         params: GeneratedDatasetParameters,
#         **kwargs
#     ) -> DataSet:
#     ...
#


def background_exp(
        number_of_dimensions: int,
        number_of_background_events: int,
        shape_nuisance_value: float = 0,
        **kwargs
) -> DataSet:
    return DataSet(np.random.exponential(
        scale=np.exp(shape_nuisance_value),
        size=(number_of_background_events, number_of_dimensions),
    ))

FLOAT_OR_ARRAY = TypeVar('FLOAT_OR_ARRAY', bound=Union[float, np.ndarray])
def decaying_exp(
        x: FLOAT_OR_ARRAY,
) -> FLOAT_OR_ARRAY:
    return np.exp(-x)
