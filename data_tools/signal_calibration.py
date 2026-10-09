"""Calibrate generated signal yields to a continuous injected significance."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from data_tools.detector.analytic_efficiency import generated_detector_efficiency
from data_tools.event_generation import background, signal
from data_tools.event_generation.distribution import (
    GeneratorSelectionConfig,
    normalize_generator_selection,
    resolve_generator,
)
from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_continuous,
)
from data_tools.signal_calibration_config import (
    SignalCalibrationConfig,
    load_signal_calibration_config,
)


def calc_n_signal_events_for_target_injected_t_significance(
    background_pdf: Callable[[float | np.ndarray], float | np.ndarray],
    signal_pdf: Callable[[float | np.ndarray], float | np.ndarray],
    n_background_events: int,
    target_significance: float,
    upper_limit: float | np.ndarray = np.inf,
    max_n_signal_events: float = 1e9,
    detector_efficiency: Callable | None = None,
) -> float:
    """Solve the continuous injected-significance equation for signal yield.

    The returned signal yield ``n_signal_events`` satisfies, up to the
    integration tolerance,

    ``calc_injected_t_significance_by_sqrt_q0_continuous(...) == target_significance``.

    The PDFs, background yield, integration limits, and optional detector
    efficiency are passed unchanged to the forward calculation used by the
    plotting code.
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
            detector_efficiency=detector_efficiency,
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
    upper_limit: float | np.ndarray | None = None,
    max_n_signal_events: float = 1e9,
    detector_efficiency: Callable | None = None,
) -> float:
    """Calibrate dataset generator specifications to a target significance.

    The generator specifications use the same ``function`` and ``arguments``
    objects as dataset configuration. If ``upper_limit`` is omitted, the
    component-wise maximum of the two generator integration limits is used.
    Without an efficiency callable, this explicit-PDF mode is generated-level.
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
        detector_efficiency=detector_efficiency,
    )


def calc_n_signal_events_for_config(
    config: SignalCalibrationConfig,
    target_significance: float,
    upper_limit: float | np.ndarray | None = None,
    max_n_signal_events: float = 1e9,
) -> float:
    """Invert the plotter's detector-level significance using configured inputs."""
    parameters = config.signal_dataset_parameters
    return calc_n_signal_events_for_target_injected_t_significance(
        background_pdf=parameters.dataset_generated__background_pdf,
        signal_pdf=parameters.dataset_generated__signal_pdf,
        n_background_events=parameters.dataset__mean_number_of_background_events,
        target_significance=target_significance,
        upper_limit=(
            parameters.dataset_generated__integration_upper_limits
            if upper_limit is None
            else upper_limit
        ),
        max_n_signal_events=max_n_signal_events,
        detector_efficiency=generated_detector_efficiency(config, parameters),
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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        help="Ordered generated-dataset and detector configurations; includes nominal detector acceptance.",
    )
    mode.add_argument(
        "--background-generator",
        type=_parse_generator,
        help="JSON generator specification used for the background PDF.",
    )
    parser.add_argument(
        "--signal-generator",
        type=_parse_generator,
        help="JSON generator specification used for the signal PDF.",
    )
    parser.add_argument("--number-of-dimensions", type=int)
    parser.add_argument("--background-events", type=int)
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

    manual_fields = (
        arguments.signal_generator,
        arguments.number_of_dimensions,
        arguments.background_events,
    )
    if arguments.configs:
        if any(value is not None for value in manual_fields):
            parser.error(
                "--configs cannot be combined with explicit generator dimensions or yields"
            )
        config = load_signal_calibration_config(arguments.configs)
        dimensions = (
            config.signal_dataset_parameters.dataset_generated__number_of_dimensions
        )
    else:
        if any(value is None for value in manual_fields):
            parser.error(
                "Explicit mode requires --signal-generator, --number-of-dimensions, and --background-events"
            )
        dimensions = arguments.number_of_dimensions
    upper_limit = _parse_upper_limit(arguments.upper_limit, dimensions)

    results = []
    for target_significance in arguments.target_significance:
        if arguments.configs:
            count = calc_n_signal_events_for_config(
                config, target_significance, upper_limit, arguments.max_signal_events
            )
        else:
            count = calc_n_signal_events_for_generated_signal(
                background_generator=arguments.background_generator,
                signal_generator=arguments.signal_generator,
                number_of_dimensions=dimensions,
                n_background_events=arguments.background_events,
                target_significance=target_significance,
                upper_limit=upper_limit,
                max_n_signal_events=arguments.max_signal_events,
            )
        results.append(
            {"target_significance": target_significance, "mean_signal_events": count}
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
