from typing import Callable, Union
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import erfinv
from scipy.stats import norm, chi2
from warnings import simplefilter


def calc_t_test_statistic(tau: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """
    Calculate the test statistic t from the tau value
    """
    return -2 * tau


def calc_t_significance_by_chi2_percentile(
          t_distribution: np.ndarray,
          degrees_of_freedom: int,
) -> float:
    return norm.ppf(
         chi2.cdf(np.median(t_distribution), df=degrees_of_freedom)
    )


def calc_t_significance_relative_to_background(
        t_value: np.float64,
        background_only_t_values: np.ndarray,
):
    """
    Calculate the significance (Z-score) of the observed t values
    relative to the null hypothesis t values.
    """ 
    num_background_lower_t_values = np.count_nonzero(background_only_t_values <= t_value)
    fraction_lower_background_t_values = num_background_lower_t_values / len(background_only_t_values)
    stretched_fraction_lower_background_t_values = fraction_lower_background_t_values * 2 - 1
    z_score = np.sqrt(2) * erfinv(stretched_fraction_lower_background_t_values)
    return z_score


def calc_median_t_significance_relative_to_background(
        background_only_t_values: np.ndarray,
        signal_t_values: np.ndarray,
) -> float:
    """
    Use the median of the signal t value distribution to estimate its
    significance relative to the null hypothesis.
    """
    return calc_t_significance_relative_to_background(
        np.median(signal_t_values),
        background_only_t_values
    )


def calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf: Callable[[float], float],
        signal_pdf: Callable[[float], float],
        n_background_events: int,
        n_signal_events: int,
        upper_limit: float = np.inf,
):
    """
    Calculate significance by formula (33) from our Symmetrized Approach paper.

    The method is integrating over analytic signal and background pdfs instead
    of bin-wise.

    Upper limit is needed especially for long tials, say, decaying exponentials
    division.
    """
    
    integrand = lambda x: (
        n_signal_events * signal_pdf(x) + n_background_events * background_pdf(x)
    ) * np.log(
        1 + n_signal_events * signal_pdf(x) / (n_background_events * background_pdf(x))
    )

    try:
        # Convert warnings to exceptions
        simplefilter("error", IntegrationWarning)
        integral, _ = quad(integrand, 0, upper_limit)
    except IntegrationWarning:
         raise ValueError(f"Integration unsuccessful, try reducing upper limit from {upper_limit}")
    finally:
        # Reset warning filter to default
        simplefilter("default", IntegrationWarning)
    
    q0 = 2 * (-n_signal_events + integral)

    return np.sqrt(q0)


def calc_injected_t_significance_by_sqrt_q0_binned(
        background_t_distribution: np.ndarray,
        signal_t_distribution: np.ndarray,
        n_signal_events: int,
        background_fraction: float,
):
        q0 = 2 * (-n_signal_events + np.sum(
             (mu * data + (bkg * background_fraction)) * \
                np.log(mu * data / (bkg * background_fraction) + 1)
        ))
        return np.sqrt(q0)
