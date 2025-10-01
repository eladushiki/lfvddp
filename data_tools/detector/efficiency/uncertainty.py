from typing import Callable
from scipy.stats import truncnorm
import numpy as np

from data_tools.detector.constants import TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD

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
#
# Important: assume you get the binned data as input. It is the developer responsibility
# to apply consistency within bins if needed.


def detector_uncertainty_no_uncertainty(
        detector_efficiency: Callable[[np.ndarray], np.ndarray],
):
    return detector_efficiency


def detector_uncertainty_10_percent_constant_diminish(
        detector_efficiency: Callable[[np.ndarray], np.ndarray],
):
    def uncertainty_wrapper(binned_x: np.ndarray) -> np.ndarray:
        clean_efficiency = detector_efficiency(binned_x)
        defected_efficiency = clean_efficiency * 0.9
        return defected_efficiency
    
    return uncertainty_wrapper


def detector_uncertainty_gaussian_noise(
        detector_efficiency: Callable[[np.ndarray], np.ndarray],
):
    """
    Adding a gaussian distributed error to the detector efficiency **by bin**.
    Using truncated normal distribution to prevent weird edge effects
    """
    def uncertainty_wrapper(binned_x: np.ndarray) -> np.ndarray:
        unique_bins, unique_inverse = np.unique(binned_x, axis=0, return_inverse=True)
        unique_efficiency = detector_efficiency(unique_bins)
        
        relative_error_magnitude_max = 0.5
        relative_error_std = TYPICAL_DETECTOR_BIN_UNCERTAINTY_STD
        relative_errors = truncnorm.rvs(
            -relative_error_magnitude_max,
            relative_error_magnitude_max,
            loc=0,
            scale=relative_error_std,
            size=unique_efficiency.shape
        )
        errored_unique_efficiency = unique_efficiency * (1 + relative_errors)
        errored_efficiency = errored_unique_efficiency[unique_inverse]
        return errored_efficiency
    
    return uncertainty_wrapper
