"""Calibrate generated signal yields to a continuous injected significance."""

import argparse
import json
from typing import Callable, Union

import numpy as np
from scipy.optimize import brentq

from data_tools.event_generation import background, signal
from data_tools.event_generation.distribution import (
    GeneratorSelectionConfig,
    normalize_generator_selection,
    resolve_generator,
)
from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_continuous,
)


def calc_n_signal_events_for_target_injected_t_significance(
    background_pdf: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    signal_pdf: Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]],
    n_background_events: int,
    target_significance: float,
    upper_limit: Union[float, np.ndarray] = np.inf,
    max_n_signal_events: float = 1e9,
) -> float:
    """Solve the continuous injected-significance equation for signal yield.

    The returned signal yield ``n_signal_events`` satisfies, up to the
    integration tolerance,

    ``calc_injected_t_significance_by_sqrt_q0_continuous(...) == target_significance``.

    The PDFs, background yield, and integration limits are passed unchanged to
    the forward calculation used by the plotting code.
    """
    if n_background_events <= 0:
        raise ValueError("n_background_events must be positive")
    if not np.isfinite(target_significance) or target_significance < 0:
        raise ValueError("target_significance must be a finite non-negative number")
    if not np.isfinite(max_n_signal_events) or max_n_signal_events <= 0:
        raise ValueError("max_n_signal_events must be a finite positive number")
    if target_significance == 0:
        return 0.0

    target_q0 = target_significance**2

    def q0_minus_target(n_signal_events: float) -> float:
        significance = calc_injected_t_significance_by_sqrt_q0_continuous(
            background_pdf=background_pdf,
            signal_pdf=signal_pdf,
            n_background_events=n_background_events,
            n_signal_events=n_signal_events,
            upper_limit=upper_limit,
        )
        return significance**2 - target_q0

    upper_bracket = min(1.0, max_n_signal_events)
    while q0_minus_target(upper_bracket) < 0:
        upper_bracket *= 2
        if upper_bracket > max_n_signal_events:
            raise ValueError(
                "Target significance was not reached below "
                f"max_n_signal_events={max_n_signal_events:g}"
            )

    return float(brentq(q0_minus_target, a=0.0, b=upper_bracket))


def calc_n_signal_events_for_generated_signal(
    background_generator: GeneratorSelectionConfig,
    signal_generator: GeneratorSelectionConfig,
    number_of_dimensions: int,
    n_background_events: int,
    target_significance: float,
    upper_limit: Union[float, np.ndarray, None] = None,
    max_n_signal_events: float = 1e9,
) -> float:
    """Calibrate dataset generator specifications to a target significance.

    The generator specifications use the same ``function`` and ``arguments``
    objects as dataset configuration. If ``upper_limit`` is omitted, the
    component-wise maximum of the two generator integration limits is used.
    """
    background_distribution = resolve_generator(
        background,
        normalize_generator_selection(background_generator),
        number_of_dimensions,
    )
    signal_distribution = resolve_generator(
        signal,
        normalize_generator_selection(signal_generator),
        number_of_dimensions,
    )
    if upper_limit is None:
        upper_limit = np.maximum(
            background_distribution.integration_upper_limits,
            signal_distribution.integration_upper_limits,
        )
        if number_of_dimensions == 1:
            upper_limit = upper_limit.item()

    return calc_n_signal_events_for_target_injected_t_significance(
        background_pdf=background_distribution.pdf,
        signal_pdf=signal_distribution.pdf,
        n_background_events=n_background_events,
        target_significance=target_significance,
        upper_limit=upper_limit,
        max_n_signal_events=max_n_signal_events,
    )


def _parse_generator(value: str) -> GeneratorSelectionConfig:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            f"generator must be valid JSON: {error.msg}"
        ) from error
    if not isinstance(parsed, (dict, list)):
        raise argparse.ArgumentTypeError(
            "generator JSON must be an object or a list of objects"
        )
    return parsed


def _parse_upper_limit(
    values: list[float] | None,
    number_of_dimensions: int,
) -> float | np.ndarray | None:
    if values is None:
        return None
    if len(values) == 1:
        if number_of_dimensions == 1:
            return values[0]
        return np.full(number_of_dimensions, values[0])
    if len(values) != number_of_dimensions:
        raise ValueError("--upper-limit accepts one value or one value per dimension")
    return np.asarray(values, dtype=float)


def main(argv: list[str] | None = None) -> None:
    """Print calibrated signal yields for one or more target significances."""
    parser = argparse.ArgumentParser(
        description=(
            "Solve the continuous injected-significance equation for the "
            "mean signal event count used in a generated configuration."
        )
    )
    parser.add_argument(
        "--background-generator",
        required=True,
        type=_parse_generator,
        help="JSON generator specification used for the background PDF.",
    )
    parser.add_argument(
        "--signal-generator",
        required=True,
        type=_parse_generator,
        help="JSON generator specification used for the signal PDF.",
    )
    parser.add_argument("--number-of-dimensions", required=True, type=int)
    parser.add_argument("--background-events", required=True, type=int)
    parser.add_argument(
        "--target-significance",
        required=True,
        type=float,
        nargs="+",
        help="One or more target sqrt(q0) values.",
    )
    parser.add_argument(
        "--upper-limit",
        type=float,
        nargs="+",
        help="One common limit or one limit per observable dimension.",
    )
    parser.add_argument("--max-signal-events", type=float, default=1e9)
    arguments = parser.parse_args(argv)

    upper_limit = _parse_upper_limit(
        arguments.upper_limit,
        arguments.number_of_dimensions,
    )
    results = [
        {
            "target_significance": target_significance,
            "mean_signal_events": calc_n_signal_events_for_generated_signal(
                background_generator=arguments.background_generator,
                signal_generator=arguments.signal_generator,
                number_of_dimensions=arguments.number_of_dimensions,
                n_background_events=arguments.background_events,
                target_significance=target_significance,
                upper_limit=upper_limit,
                max_n_signal_events=arguments.max_signal_events,
            ),
        }
        for target_significance in arguments.target_significance
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
