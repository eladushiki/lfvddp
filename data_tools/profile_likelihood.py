from math import fsum
from typing import Callable, Union
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import erfinv
from scipy.stats import norm, chi2
from warnings import catch_warnings, simplefilter


_MAX_QUADRATURE_INTERVAL_WIDTH = 1.0
_QUADRATURE_SUBDIVISION_LIMIT = 200


def calc_t_test_statistic_NPLM(tau: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """
    Calculate the test statistic t from the tau value
    """
    return -2 * tau


def calc_t_LFVDDP(
    numerator: Union[int, float, np.ndarray],
    denominator: Union[int, float, np.ndarray],
) -> Union[int, float, np.ndarray]:
    """Calculate t from the two independently minimized expressions."""
    return -2 * numerator + 2 * denominator


def calc_t_significance_by_chi2_percentile(
          t_distribution: np.ndarray,
          degrees_of_freedom: int,
) -> float:
    return norm.ppf(
         chi2.cdf(np.median(t_distribution), df=degrees_of_freedom)
    )


def calc_t_significance_by_gaussian_fit_percentile(
          background_only_distribution: np.ndarray,
          t_value: np.float64,
          n_bins: int = 100,
) -> float:
    # Fit a gaussian to the background-only t distribution
    mu, std = norm.fit(background_only_distribution)

    # Estimate significance of the t value
    return (t_value - mu) / std


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

    Upper limit is needed especially for long tails, say, decaying exponentials
    division.
    """
    if n_signal_events <= 0:
         return 0

    def integrand(x: float) -> float:
        signal_rate_density = n_signal_events * signal_pdf(x)
        background_rate_density = n_background_events * background_pdf(x)

        # Both PDFs can underflow to zero in a sufficiently remote tail. The
        # limiting contribution there is zero, whereas evaluating the original
        # expression directly produces 0 / 0 and poisons the quadrature.
        if signal_rate_density == 0:
            return 0.0
        if background_rate_density == 0:
            return np.inf

        return (
            signal_rate_density + background_rate_density
        ) * np.log1p(signal_rate_density / background_rate_density)

    if np.isfinite(upper_limit):
        interval_boundaries = np.arange(
            0,
            upper_limit,
            _MAX_QUADRATURE_INTERVAL_WIDTH,
        )
        interval_boundaries = np.append(interval_boundaries, upper_limit)
    else:
        interval_boundaries = np.array([0, upper_limit])

    try:
        with catch_warnings():
            simplefilter("error", IntegrationWarning)
            integral = fsum(
                quad(
                    integrand,
                    lower_bound,
                    upper_bound,
                    limit=_QUADRATURE_SUBDIVISION_LIMIT,
                )[0]
                for lower_bound, upper_bound in zip(
                    interval_boundaries[:-1],
                    interval_boundaries[1:],
                )
                if lower_bound < upper_bound
            )
    except IntegrationWarning as warning:
         raise ValueError(
             f"Integration unsuccessful up to upper limit {upper_limit}"
         ) from warning
    
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
