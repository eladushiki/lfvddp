from typing import Callable
from scipy.stats import truncnorm
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
    """
    Using truncated normal distribution to prevent weird edge effects
    """
    def uncertainty_wrapper(x: np.ndarray) -> np.ndarray:
        clean_efficiency = detector_efficiency(x)
        
        relative_error_magnitude_max = 0.5
        relative_error_std = 0.2
        relative_errors = truncnorm.rvs(
            -relative_error_magnitude_max,
            relative_error_magnitude_max,
            loc=0,
            scale=relative_error_std,
            size=clean_efficiency.shape
        )
        errors = clean_efficiency * relative_errors

        defected_efficiency = clean_efficiency + errors
        return defected_efficiency
    
    return uncertainty_wrapper
