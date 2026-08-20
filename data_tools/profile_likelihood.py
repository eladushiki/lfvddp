from math import fsum
from typing import Callable, Union
import numpy as np
from scipy.integrate import IntegrationWarning, cubature, nquad
from scipy.special import erfinv, kl_div, rel_entr
from scipy.stats import norm, chi2
from warnings import catch_warnings, simplefilter, warn


_MAX_QUADRATURE_INTERVAL_WIDTH = 1.0
_QUADRATURE_SUBDIVISION_LIMIT = 200
_QUADRATURE_ABSOLUTE_TOLERANCE = 1e-3
_QUADRATURE_RELATIVE_TOLERANCE = 1e-4
_CUBATURE_RULE = "genz-malik"
_CUBATURE_MAX_SUBDIVISIONS = 10_000
_CUBATURE_ABSOLUTE_TOLERANCE = 1e-3
_CUBATURE_RELATIVE_TOLERANCE = 5e-3


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


def _normalize_integration_upper_limits(
        upper_limit: Union[float, np.ndarray],
) -> np.ndarray:
    """Return one positive upper bound for each observable dimension."""
    upper_limits = np.asarray(upper_limit, dtype=float)
    if upper_limits.ndim == 0:
        upper_limits = upper_limits.reshape(1)
    if upper_limits.ndim != 1 or upper_limits.size == 0:
        raise ValueError("upper_limit must be a non-empty scalar or 1-D array")
    if np.any(np.isnan(upper_limits)) or np.any(upper_limits <= 0):
        raise ValueError("upper_limit values must be positive numbers or infinity")
    return upper_limits


def _pdf_density_at_coordinates(
        pdf: Callable[[Union[float, np.ndarray]], float],
        coordinates: tuple[float, ...],
) -> float:
    """Evaluate a PDF at one point and validate its scalar density."""
    evaluation_point = coordinates[0] if len(coordinates) == 1 else np.asarray(coordinates)
    density = np.asarray(pdf(evaluation_point))
    if density.ndim != 0 or not np.isfinite(density):
        raise ValueError("PDF must return a finite scalar density")
    if density < 0:
        raise ValueError("PDF must return a non-negative density")
    return float(density)


def _pdf_densities_at_points(
        pdf: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
        points: np.ndarray,
) -> np.ndarray:
    """Evaluate a PDF on a batch, falling back to its scalar point contract."""
    try:
        densities = np.asarray(pdf(points))
    except (AssertionError, IndexError, TypeError, ValueError):
        densities = np.empty(0)

    if densities.ndim == 0:
        densities = np.full(points.shape[0], densities.item())
    elif densities.shape != (points.shape[0],):
        densities = np.asarray([
            _pdf_density_at_coordinates(pdf, tuple(point))
            for point in points
        ])

    if not np.all(np.isfinite(densities)):
        raise ValueError("PDF must return finite scalar densities")
    if np.any(densities < 0):
        raise ValueError("PDF must return non-negative densities")
    return densities.astype(float, copy=False)


def _one_dimensional_integration_regions(
        upper_limit: float,
) -> list[list[tuple[float, float]]]:
    """Build bounded-width regions for the legacy 1D quadrature path."""
    if not np.isfinite(upper_limit):
        return [[(0.0, np.inf)]]

    boundaries = np.append(
        np.arange(0, upper_limit, _MAX_QUADRATURE_INTERVAL_WIDTH),
        upper_limit,
    )
    return [
        [(lower_bound, upper_bound)]
        for lower_bound, upper_bound in zip(boundaries[:-1], boundaries[1:])
        if lower_bound < upper_bound
    ]


def calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf: Callable[
            [Union[float, np.ndarray]], Union[float, np.ndarray]
        ],
        signal_pdf: Callable[
            [Union[float, np.ndarray]], Union[float, np.ndarray]
        ],
        n_background_events: int,
        n_signal_events: int,
        upper_limit: Union[float, np.ndarray] = np.inf,
):
    """Calculate formula (32) from 2024 paper, significance for distributions
    over one or more observables with known pdfs.

    A scalar ``upper_limit`` defines the existing one-dimensional domain
    ``[0, upper_limit]``. A one-dimensional array supplies one upper bound per
    observable; multidimensional PDF callables receive a coordinate array in
    that same observable order. Multidimensional limits must be finite.
    """
    if n_signal_events <= 0:
        return 0

    upper_limits = _normalize_integration_upper_limits(upper_limit)

    def integrand(*coordinates: float) -> float:
        signal_rate_density = n_signal_events * _pdf_density_at_coordinates(
            signal_pdf, coordinates
        )
        background_rate_density = n_background_events * _pdf_density_at_coordinates(
            background_pdf, coordinates
        )
        return rel_entr(  # = a * log(a/b)
            signal_rate_density + background_rate_density,
            background_rate_density,
        )

    if upper_limits.size > 1:
        if not np.all(np.isfinite(upper_limits)):
            raise ValueError(
                "Multidimensional integration upper limits must be finite"
            )

        def q0_integrand(points: np.ndarray) -> np.ndarray:
            signal_rate_density = n_signal_events * _pdf_densities_at_points(
                signal_pdf, points
            )
            background_rate_density = (
                n_background_events
                * _pdf_densities_at_points(background_pdf, points)
            )
            return 2 * kl_div(
                signal_rate_density + background_rate_density,
                background_rate_density,
            )

        result = cubature(
            q0_integrand,
            np.zeros(upper_limits.size),
            upper_limits,
            rule=_CUBATURE_RULE,
            rtol=_CUBATURE_RELATIVE_TOLERANCE,
            atol=_CUBATURE_ABSOLUTE_TOLERANCE,
            max_subdivisions=_CUBATURE_MAX_SUBDIVISIONS,
        )
        q0 = np.asarray(result.estimate).item()
        estimated_error = np.asarray(result.error).item()
        if not np.isfinite(q0) or not np.isfinite(estimated_error):
            raise ValueError(
                "Multidimensional significance integration was non-finite"
            )
        if result.status != "converged":
            warn(
                "Multidimensional significance reached its cubature subdivision "
                f"cap with estimated error {estimated_error:g}",
                RuntimeWarning,
                stacklevel=2,
            )
    else:
        try:
            with catch_warnings():
                simplefilter("error", IntegrationWarning)
                integral = fsum(
                    nquad(
                        integrand,
                        interval_bounds,
                        opts={
                            "limit": _QUADRATURE_SUBDIVISION_LIMIT,
                            "epsabs": _QUADRATURE_ABSOLUTE_TOLERANCE,
                            "epsrel": _QUADRATURE_RELATIVE_TOLERANCE,
                        },
                    )[0]
                    for interval_bounds in _one_dimensional_integration_regions(
                        upper_limits.item()
                    )
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
