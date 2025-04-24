import numpy as np

## DEVELOPER NOTE: This is a special namespace, functions defined
# here are automatically added and can be called as detector
# effects.
# Have the function signature as:f
#
# def detector_efficiency_...(
#   x: np.ndarray
# ) -> np.ndarray:
#   ...
#
# Return an array of the same shape as x with the probability of
# inclusion in the final dataset, between 0 and 1. Any other value
# would be truncated.


## Possible detector efficiency shapes
def detector_efficiency_tanh(x: np.ndarray) -> np.ndarray:
    efficiency = (np.tanh(x[:, 0]) + 1) / 2
    return efficiency.flatten()


## Default for any type could be
def detector_unaffected(x: np.ndarray) -> np.ndarray:
    efficiency = np.ones((x.shape[0], 1))
    return efficiency.flatten()
