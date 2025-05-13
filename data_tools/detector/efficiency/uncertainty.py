from typing import Callable
import numpy as np

# Uncertainty variations for detector effect efficiencies
# Each function implemented here would be recognized and recalled by name
# Have the function signature as:
#
DETECTOR_EFFICIENCY_UNCERTAINTY_TYPE = Callable[[Callable[[np.ndarray], np.ndarray]], Callable[[np.ndarray], np.ndarray]]
# def detector_efficiency_...(
#   detector_efficiency: Callable[[np.ndarray], np.ndarray]
# ) -> Callable[[np.ndarray], np.ndarray]:
#   ...
#
# Should be implemented as a wrapper of the original efficiency


def detector_uncertainty_no_uncertainty(
        detector_efficiency: Callable[[np.ndarray], np.ndarray],
):
    return detector_efficiency


def detector_uncertainty_10_percent_constant_diminish(
        detector_efficiency: Callable[[np.ndarray], np.ndarray],
):
    def uncertainty_wrapper(x: np.ndarray) -> np.ndarray:
        clean_efficiency = detector_efficiency(x)
        defected_efficiency = clean_efficiency * 0.9
        return defected_efficiency
    
    return uncertainty_wrapper


def detector_uncertainty_gaussian_noise(
        detector_efficiency: Callable[[np.ndarray], np.ndarray],
):
    def uncertainty_wrapper(x: np.ndarray) -> np.ndarray:
        clean_efficiency = detector_efficiency(x)
        defected_efficiency = clean_efficiency + np.random.normal(0, 0.1, size=clean_efficiency.shape)
        return defected_efficiency
    
    return uncertainty_wrapper
