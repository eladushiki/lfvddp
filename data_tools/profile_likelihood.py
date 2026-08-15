from itertools import product
from math import fsum
from typing import Callable, Union
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import IntegrationWarning, nquad
from scipy.special import erfinv
from scipy.stats import norm, chi2
from warnings import catch_warnings, simplefilter


_MAX_QUADRATURE_INTERVAL_WIDTH = 1.0
_QUADRATURE_SUBDIVISION_LIMIT = 200
_ND_GAUSS_LEGENDRE_ORDER = 8
_ND_GAUSS_LEGENDRE_NODES, _ND_GAUSS_LEGENDRE_WEIGHTS = leggauss(
    _ND_GAUSS_LEGENDRE_ORDER
)


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


def _integration_intervals(upper_limits: np.ndarray) -> list[list[tuple[float, float]]]:
    """Split finite integration dimensions for reliable narrow-PDF quadrature."""
    intervals = []
    for upper_limit in upper_limits:
        if np.isfinite(upper_limit):
            boundaries = np.append(
                np.arange(0, upper_limit, _MAX_QUADRATURE_INTERVAL_WIDTH),
                upper_limit,
            )
            intervals.append([
                (lower_bound, upper_bound)
                for lower_bound, upper_bound in zip(boundaries[:-1], boundaries[1:])
                if lower_bound < upper_bound
            ])
        else:
            intervals.append([(0.0, np.inf)])
    return intervals


def _pdf_densities_at_coordinates(
    pdf: Callable[[Union[float, np.ndarray]], float],
    coordinates: np.ndarray,
) -> np.ndarray:
    """Evaluate a PDF over a coordinate batch, using scalar compatibility when needed."""
    if getattr(pdf, "supports_batch_evaluation", False):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            densities = np.asarray(pdf(coordinates))
    else:
        densities = np.asarray([
            _pdf_density_at_coordinates(pdf, tuple(point))
            for point in coordinates
        ])

    if densities.shape != (coordinates.shape[0],):
        raise ValueError("PDF must return one scalar density per coordinate point")
    if not np.all(np.isfinite(densities)):
        raise ValueError("PDF must return finite scalar densities")
    if np.any(densities < 0):
        raise ValueError("PDF must return non-negative densities")
    return densities


def _asimov_integrand_values(
    signal_rate_density: Union[float, np.ndarray],
    background_rate_density: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """Evaluate the continuous Asimov integrand for scalar or batched rates."""
    signal_rates, background_rates = np.broadcast_arrays(
        signal_rate_density,
        background_rate_density,
    )
    values = np.zeros(signal_rates.shape)
    signal_is_positive = signal_rates != 0
    background_is_zero = background_rates == 0
    values[signal_is_positive & background_is_zero] = np.inf
    regular = signal_is_positive & ~background_is_zero
    values[regular] = (
        signal_rates[regular] + background_rates[regular]
    ) * np.log1p(signal_rates[regular] / background_rates[regular])
    return values.item() if values.ndim == 0 else values


def _integrate_nd_interval(
    background_pdf: Callable[[Union[float, np.ndarray]], float],
    signal_pdf: Callable[[Union[float, np.ndarray]], float],
    n_background_events: int,
    n_signal_events: int,
    interval_bounds: tuple[tuple[float, float], ...],
) -> float:
    """Integrate one finite N-D interval with tensor Gauss-Legendre quadrature."""
    nodes = []
    weights = []
    for lower_bound, upper_bound in interval_bounds:
        midpoint = (lower_bound + upper_bound) / 2
        half_width = (upper_bound - lower_bound) / 2
        nodes.append(midpoint + half_width * _ND_GAUSS_LEGENDRE_NODES)
        weights.append(half_width * _ND_GAUSS_LEGENDRE_WEIGHTS)

    coordinate_grids = np.meshgrid(*nodes, indexing="ij")
    weight_grids = np.meshgrid(*weights, indexing="ij")
    coordinates = np.stack(coordinate_grids, axis=-1).reshape(
        -1,
        len(interval_bounds),
    )
    point_weights = np.prod(np.stack(weight_grids, axis=-1), axis=-1).reshape(-1)
    signal_rates = n_signal_events * _pdf_densities_at_coordinates(
        signal_pdf,
        coordinates,
    )
    background_rates = n_background_events * _pdf_densities_at_coordinates(
        background_pdf,
        coordinates,
    )
    return float(np.sum(
        point_weights * _asimov_integrand_values(signal_rates, background_rates)
    ))


def calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf: Callable[[Union[float, np.ndarray]], float],
        signal_pdf: Callable[[Union[float, np.ndarray]], float],
        n_background_events: int,
        n_signal_events: int,
        upper_limit: Union[float, np.ndarray] = np.inf,
):
    """Calculate formula (33) significance for PDFs over one or more observables.

    A scalar ``upper_limit`` defines the existing one-dimensional domain
    ``[0, upper_limit]``. A one-dimensional array supplies one upper bound per
    observable; multidimensional PDF callables receive a coordinate array in
    that same observable order.
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

        return _asimov_integrand_values(
            signal_rate_density,
            background_rate_density,
        )

    try:
        with catch_warnings():
            simplefilter("error", IntegrationWarning)
            interval_bounds = product(*_integration_intervals(upper_limits))
            if upper_limits.size == 1 or np.any(~np.isfinite(upper_limits)):
                integral = fsum(
                    nquad(
                        integrand,
                        bounds,
                        opts={"limit": _QUADRATURE_SUBDIVISION_LIMIT},
                    )[0]
                    for bounds in interval_bounds
                )
            else:
                integral = fsum(
                    _integrate_nd_interval(
                        background_pdf,
                        signal_pdf,
                        n_background_events,
                        n_signal_events,
                        tuple(bounds),
                    )
                    for bounds in interval_bounds
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
