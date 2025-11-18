from typing import Callable
import numpy as np
import pandas as pd

## DEVELOPER NOTE: This is a special namespace, functions defined
# here are automatically added and can be called as detector
# efficiencies.
# Have the function signature as:
#
DETECTOR_EFFICIENCY_TYPE = Callable[[pd.DataFrame], np.ndarray]
# def detector_efficiency_...(
#   x: pd.DataFrame
# ) -> np.ndarray:
#   ...
#
# Return an array of the same shape as x with the probability of
# inclusion in the final dataset, between 0 and 1. Any other value
# would be truncated.


## Default for any type could be
def detector_efficiency_perfect_efficiency(x: pd.DataFrame) -> np.ndarray:
    efficiency = np.ones((x.shape[0], 1))
    return efficiency.flatten()


## Possible detector efficiency shapes
def detector_efficiency_tanh(x: pd.DataFrame) -> np.ndarray:
    efficiency = (np.tanh(x[x.columns[0]] / 5) + 1) / 2
    return np.array(efficiency)
