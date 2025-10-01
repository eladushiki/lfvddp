from typing import Callable
import numpy as np

# All implemented function would automatically be recognized as
# detector error effects.
# Have the function signature as:
#
DETECTOR_ERROR_TYPE = Callable[[np.ndarray], np.ndarray]
# def detector_error_...(
#   x: np.ndarray
# ) -> np.ndarray:
#   ...
#
# Return an array of the same shape as x with the error to be
# added, cell-wise, to the dataset.


def detector_no_error(x: np.ndarray) -> np.ndarray:
    """
    No error effect, return the input array as is.
    """
    return np.zeros_like(x)
