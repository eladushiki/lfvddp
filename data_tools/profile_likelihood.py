from itertools import product
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
_MULTIDIMENSIONAL_INTERVALS_PER_AXIS = 50
_MAX_MULTIDIMENSIONAL_INTEGRATION_REGIONS = 10_000
_TENSOR_QUADRATURE_MIN_DIMENSIONS = 2
_TENSOR_QUADRATURE_TARGET_NODES_PER_AXIS = 24
_TENSOR_QUADRATURE_MAX_POINTS = 500_000
_TENSOR_QUADRATURE_MAX_BATCH_POINTS = 100_000


def calc_t_test_statistic_NPLM(
    tau: Union[int, float, np.ndarray],
) -> Union[int, float, np.ndarray]:
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


def calc_median_t_significance_by_chi2_percentile(
    t_distribution: np.ndarray,
    degrees_of_freedom: int,
) -> float:
    return norm.ppf(chi2.cdf(np.median(t_distribution), df=degrees_of_freedom))


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
    num_background_lower_t_values = np.count_nonzero(
        background_only_t_values <= t_value
    )
    fraction_lower_background_t_values = num_background_lower_t_values / len(
        background_only_t_values
    )
    stretched_fraction_lower_background_t_values = (
        fraction_lower_background_t_values * 2 - 1
    )
    z_score = np.sqrt(2) * erfinv(stretched_fraction_lower_background_t_values)
    return z_score


def calc_median_t_significance_relative_to_background(
    background_only_t_values: np.ndarray,
    signal_t_values: np.ndarray,
) -> float:
    """Estimate signal significance from its median t value under the null."""
    return calc_t_significance_relative_to_background(
        np.median(signal_t_values), background_only_t_values
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
    evaluation_point = (
        coordinates[0] if len(coordinates) == 1 else np.asarray(coordinates)
    )
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
        densities = np.asarray(
            [_pdf_density_at_coordinates(pdf, tuple(point)) for point in points]
        )

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


def _multidimensional_integration_regions(
    upper_limits: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a multidimensional domain into scale-relative cubature regions."""
    intervals_per_axis = min(
        _MULTIDIMENSIONAL_INTERVALS_PER_AXIS,
        int(_MAX_MULTIDIMENSIONAL_INTEGRATION_REGIONS ** (1 / upper_limits.size)),
    )
    boundaries = [
        np.linspace(0, upper_limit, intervals_per_axis + 1)
        for upper_limit in upper_limits
    ]
    axis_regions = [
        list(zip(axis_boundaries[:-1], axis_boundaries[1:]))
        for axis_boundaries in boundaries
    ]
    return [
        (
            np.asarray([region[0] for region in regions]),
            np.asarray([region[1] for region in regions]),
        )
        for regions in product(*axis_regions)
    ]


def _tensor_quadrature(
    integrand: Callable[[np.ndarray], np.ndarray],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
) -> float:
    """Integrate a finite domain with bounded tensor quadrature.

    Adaptive cubature can recursively subdivide for too long during plot
    aggregation. A fixed Gauss-Legendre tensor rule keeps the cost predictable
    and lets vectorized PDFs evaluate each point cloud in a small number of
    large batches.
    """
    nodes_per_axis = min(
        _TENSOR_QUADRATURE_TARGET_NODES_PER_AXIS,
        max(2, int(_TENSOR_QUADRATURE_MAX_POINTS ** (1 / upper_limits.size))),
    )
    base_nodes, base_weights = np.polynomial.legendre.leggauss(nodes_per_axis)
    return _tensor_quadrature_with_rule(
        integrand,
        lower_limits,
        upper_limits,
        base_nodes,
        base_weights,
    )


def _weighted_integrand_sum(
    integrand: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    weights: np.ndarray,
) -> float:
    total = 0.0
    for start in range(0, points.shape[0], _TENSOR_QUADRATURE_MAX_BATCH_POINTS):
        stop = start + _TENSOR_QUADRATURE_MAX_BATCH_POINTS
        values = np.asarray(integrand(points[start:stop]), dtype=float)
        if values.shape != (points[start:stop].shape[0],) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("Tensor quadrature integrand returned non-finite values")
        total += float(np.dot(weights[start:stop], values))
    return total


def _tensor_quadrature_with_rule(
    integrand: Callable[[np.ndarray], np.ndarray],
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    base_nodes: np.ndarray,
    base_weights: np.ndarray,
) -> float:
    widths = upper_limits - lower_limits
    axis_nodes = [
        lower_limit + 0.5 * width * (base_nodes + 1)
        for lower_limit, width in zip(lower_limits, widths)
    ]
    axis_weights = [0.5 * width * base_weights for width in widths]

    coordinate_mesh = np.meshgrid(*axis_nodes, indexing="ij")
    points = np.column_stack([coordinates.ravel() for coordinates in coordinate_mesh])

    weights = axis_weights[0]
    for next_weights in axis_weights[1:]:
        weights = np.multiply.outer(weights, next_weights).ravel()

    return _weighted_integrand_sum(integrand, points, weights)


def _high_dimensional_tensor_quadrature(
    integrand: Callable[[np.ndarray], np.ndarray],
    upper_limits: np.ndarray,
) -> float:
    """Integrate a high-dimensional domain with one bounded tensor rule."""
    return _tensor_quadrature(
        integrand,
        np.zeros_like(upper_limits),
        upper_limits,
    )


def _regioned_tensor_quadrature(
    integrand: Callable[[np.ndarray], np.ndarray],
    upper_limits: np.ndarray,
) -> float:
    """Integrate a low-dimensional domain with bounded tensor rules per region."""
    nodes_per_axis = min(
        _TENSOR_QUADRATURE_TARGET_NODES_PER_AXIS,
        max(2, int(_TENSOR_QUADRATURE_MAX_POINTS ** (1 / upper_limits.size))),
    )
    base_nodes, base_weights = np.polynomial.legendre.leggauss(nodes_per_axis)
    axis_nodes = []
    axis_weights = []
    intervals_per_axis = min(
        _MULTIDIMENSIONAL_INTERVALS_PER_AXIS,
        int(_MAX_MULTIDIMENSIONAL_INTEGRATION_REGIONS ** (1 / upper_limits.size)),
    )
    for upper_limit in upper_limits:
        boundaries = np.linspace(0, upper_limit, intervals_per_axis + 1)
        nodes = []
        weights = []
        for lower_bound, upper_bound in zip(boundaries[:-1], boundaries[1:]):
            width = upper_bound - lower_bound
            nodes.append(lower_bound + 0.5 * width * (base_nodes + 1))
            weights.append(0.5 * width * base_weights)
        axis_nodes.append(np.concatenate(nodes))
        axis_weights.append(np.concatenate(weights))

    coordinate_mesh = np.meshgrid(*axis_nodes, indexing="ij")
    points = np.column_stack([coordinates.ravel() for coordinates in coordinate_mesh])

    weights = axis_weights[0]
    for next_weights in axis_weights[1:]:
        weights = np.multiply.outer(weights, next_weights).ravel()

    return _weighted_integrand_sum(integrand, points, weights)


def calc_injected_t_significance_by_sqrt_q0_continuous(
    background_pdf: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    signal_pdf: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    n_background_events: int,
    n_signal_events: int,
    upper_limit: Union[float, np.ndarray] = np.inf,
    detector_efficiency: Callable | None = None,
):
    """Calculate formula (32) from 2024 paper, significance for distributions
    over one or more observables with known pdfs.

    ``detector_efficiency`` accepts the same coordinates as the PDFs and
    multiplies both event densities. Accepted densities are not renormalized;
    the likelihood contribution includes the accepted signal subtraction.

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
        acceptance = (
            1.0
            if detector_efficiency is None
            else _pdf_density_at_coordinates(detector_efficiency, coordinates)
        )
        signal_rate_density *= acceptance
        background_rate_density *= acceptance
        return rel_entr(
            signal_rate_density + background_rate_density,
            background_rate_density,
        )

    if upper_limits.size > 1:
        if not np.all(np.isfinite(upper_limits)):
            raise ValueError("Multidimensional integration upper limits must be finite")

        def q0_integrand(points: np.ndarray) -> np.ndarray:
            signal_rate_density = n_signal_events * _pdf_densities_at_points(
                signal_pdf, points
            )
            background_rate_density = n_background_events * _pdf_densities_at_points(
                background_pdf, points
            )
            if detector_efficiency is not None:
                acceptance = _pdf_densities_at_points(detector_efficiency, points)
                signal_rate_density *= acceptance
                background_rate_density *= acceptance
            return 2 * kl_div(
                signal_rate_density + background_rate_density,
                background_rate_density,
            )

        if upper_limits.size == 2:
            q0 = _regioned_tensor_quadrature(q0_integrand, upper_limits)
        elif upper_limits.size >= _TENSOR_QUADRATURE_MIN_DIMENSIONS:
            q0 = _high_dimensional_tensor_quadrature(q0_integrand, upper_limits)
        else:
            results = [
                cubature(
                    q0_integrand,
                    lower_bounds,
                    upper_bounds,
                    rule=_CUBATURE_RULE,
                    rtol=_CUBATURE_RELATIVE_TOLERANCE,
                    atol=_CUBATURE_ABSOLUTE_TOLERANCE,
                    max_subdivisions=_CUBATURE_MAX_SUBDIVISIONS,
                )
                for lower_bounds, upper_bounds in _multidimensional_integration_regions(
                    upper_limits
                )
            ]
            q0 = sum(np.asarray(result.estimate).item() for result in results)
            estimated_error = sum(np.asarray(result.error).item() for result in results)
            if not np.isfinite(estimated_error):
                raise ValueError(
                    "Multidimensional significance integration error was non-finite"
                )
            if any(result.status != "converged" for result in results):
                warn(
                    "Multidimensional significance reached its cubature subdivision "
                    f"cap with estimated error {estimated_error:g}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        if not np.isfinite(q0):
            raise ValueError("Multidimensional significance integration was non-finite")
    else:

        def integrate_density(density):
            try:
                with catch_warnings():
                    simplefilter("error", IntegrationWarning)
                    return fsum(
                        nquad(
                            density,
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

        def accepted_signal_density(*coordinates):
            acceptance = (
                1.0
                if detector_efficiency is None
                else _pdf_density_at_coordinates(detector_efficiency, coordinates)
            )
            return (
                n_signal_events
                * _pdf_density_at_coordinates(signal_pdf, coordinates)
                * acceptance
            )

        accepted_signal_events = integrate_density(accepted_signal_density)
        q0 = 2 * (integrate_density(integrand) - accepted_signal_events)
    return np.sqrt(q0)


def calc_injected_t_significance_by_sqrt_q0_binned(
    background_bin_counts: np.ndarray,
    signal_bin_counts: np.ndarray,
) -> float:
    """Calculate injected significance from expected background and signal bins.

    ``background_bin_counts`` contains expected background counts per bin
    :math:`N_{b,i}` and ``signal_bin_counts`` contains expected signal counts
    per bin :math:`N_{s,i}`. The returned value is
    :math:`Z=\\sqrt{q_0}`, with
    :math:`q_0=2[-N_s + \\sum_i (N_{b,i}+N_{s,i})\\log((N_{b,i}+N_{s,i})/N_{b,i})]`
    and :math:`N_s=\\sum_i N_{s,i}`.
    """
    background_bin_counts = np.asarray(background_bin_counts, dtype=float)
    signal_bin_counts = np.asarray(signal_bin_counts, dtype=float)
    if background_bin_counts.shape != signal_bin_counts.shape:
        raise ValueError(
            "Background and signal bin counts must have the same shape; got "
            f"{background_bin_counts.shape} and {signal_bin_counts.shape}."
        )
    if np.any(~np.isfinite(background_bin_counts)) or np.any(
        ~np.isfinite(signal_bin_counts)
    ):
        raise ValueError("Binned significance counts must be finite.")
    if np.any(background_bin_counts < 0) or np.any(signal_bin_counts < 0):
        raise ValueError("Binned significance counts must be non-negative.")

    n_signal_events = float(np.sum(signal_bin_counts))
    if n_signal_events <= 0:
        return 0.0
    if np.any((background_bin_counts <= 0) & (signal_bin_counts > 0)):
        raise ValueError(
            "Cannot calculate finite binned significance where signal occupies "
            "a bin with zero expected background."
        )

    populated = background_bin_counts > 0
    q0 = 2 * (
        -n_signal_events
        + np.sum(
            (background_bin_counts[populated] + signal_bin_counts[populated])
            * np.log1p(signal_bin_counts[populated] / background_bin_counts[populated])
        )
    )
    if not np.isfinite(q0):
        raise ValueError("Binned significance was non-finite.")
    return np.sqrt(max(q0, 0.0))
