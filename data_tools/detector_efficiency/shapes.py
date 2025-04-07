import numpy as np

## DEVELOPER NOTE: This is a special namespace, functions defined
# here are automatically added and can be called as detector
# effects.

# Detector effects currently have two categories:
# 1. Detector efficiency
# 2. Detector error

## Possible detector efficiency shapes
def detector_efficiency_tanh(x: np.ndarray) -> np.ndarray:
    efficiency = (np.tanh(x[:, 0]) + 1) / 2
    return efficiency.flatten()


## Default for any type could be
def detector_unaffected(x: np.ndarray) -> np.ndarray:
    efficiency = np.ones((x.shape[0], 1))
    return efficiency.flatten()
