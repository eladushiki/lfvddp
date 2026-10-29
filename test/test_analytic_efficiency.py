from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from data_tools.detector.analytic_efficiency import generated_detector_efficiency
from data_tools.detector.efficiency.shapes import detector_efficiency_tanh
from data_tools.event_generation.background import ExponentialBackground
from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_continuous,
)
from frame.aggregate import ResultAggregator
from test.environment import ConfigType


@pytest.mark.parametrize("dimensions", [1, 2, 4])
@pytest.mark.parametrize("acceptance", [0.0, 0.9, 1.0])
def test_constant_acceptance_scales_significance_without_renormalization(
    dimensions, acceptance
):
    distribution = ExponentialBackground(dimensions, domain_max=5)
    arguments = {
        "background_pdf": distribution.pdf,
        "signal_pdf": distribution.pdf,
        "n_background_events": 100,
        "n_signal_events": 10,
        "upper_limit": distribution.integration_upper_limits,
    }
    baseline = calc_injected_t_significance_by_sqrt_q0_continuous(**arguments)
    detected = calc_injected_t_significance_by_sqrt_q0_continuous(
        **arguments, detector_efficiency=lambda coordinates: acceptance
    )
    assert detected == pytest.approx(np.sqrt(acceptance) * baseline, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("dimensions", [1, 2])
def test_shape_acceptance_weights_the_existing_pdfs(dimensions):
    distribution = ExponentialBackground(dimensions, domain_max=5)

    def acceptance(coordinates):
        points = np.asarray(coordinates)
        values = detector_efficiency_tanh(pd.DataFrame(np.atleast_2d(points)))
        return values.item() if points.ndim < 2 else values

    one_dimensional_pdf = ExponentialBackground(1).pdf
    accepted_mass = quad(lambda x: one_dimensional_pdf(x) * acceptance(x), 0, 5)[0]
    generated_mass = quad(one_dimensional_pdf, 0, 5)[0]
    per_unit_mass_q0 = 2 * (110 * np.log1p(10 / 100) - 10)
    expected = np.sqrt(
        per_unit_mass_q0 * accepted_mass * generated_mass ** (dimensions - 1)
    )
    detected = calc_injected_t_significance_by_sqrt_q0_continuous(
        background_pdf=distribution.pdf,
        signal_pdf=distribution.pdf,
        n_background_events=100,
        n_signal_events=10,
        upper_limit=distribution.integration_upper_limits,
        detector_efficiency=acceptance,
    )
    assert detected == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET: Path(
                f"test/configs/dataset/disjoint_{dimensions}D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR: Path(
                f"test/configs/detector/analytic_efficiency_{dimensions}D.json"
            ),
        }
        for dimensions in [1, 2]
    ]
    + [
        {
            ConfigType.DATASET: Path(
                "test/detector/configs/detector_affected_basic_ds.json"
            ),
            ConfigType.DETECTOR: Path(
                "test/detector/configs/detector_affected_basic_detector_config.json"
            ),
        }
    ],
    indirect=True,
)
def test_nominal_adapter_reuses_family_efficiency_and_observable_order(
    function_execution_context,
):
    context = function_execution_context
    dimensions = context.config.dataset_parameters[
        0
    ].dataset_generated__number_of_dimensions
    points = np.arange(1, 1 + 3 * dimensions, dtype=float).reshape(3, dimensions)
    frame = pd.DataFrame(points, columns=[f"param_{i}" for i in range(dimensions)])
    selected = frame[context.config.detector__detect_observable_names]
    for parameters in context.config.dataset_parameters:
        efficiency = generated_detector_efficiency(context, parameters)
        expected = (
            detector_efficiency_tanh(selected)
            if parameters.category.name.startswith("A")
            else np.full(
                3, 0.9 if context.config.detector__effect_b_efficiency else 1.0
            )
        )
        np.testing.assert_allclose(efficiency(points), expected)
        assert efficiency(points[0]) == pytest.approx(expected[0])


def test_aggregation_does_not_reuse_significance_across_detector_settings(
    detector_comparison_contexts, monkeypatch, tmp_path
):
    perfect, affected = detector_comparison_contexts
    parameters = perfect.config.dataset_parameters[0]
    contexts = [perfect, affected, affected]
    calls = []

    def calculation(**arguments):
        efficiency = arguments["detector_efficiency"]
        result = 1.0 if efficiency is None else efficiency(np.array([1.0, 2.0]))
        calls.append(result)
        return result

    monkeypatch.setattr(
        "frame.aggregate.ExecutionContext.discover_run_contexts",
        lambda directory: [(context, directory) for context in contexts],
    )
    monkeypatch.setattr(
        "frame.aggregate.utils__get_signal_dataset_parameters",
        lambda context: parameters,
    )
    monkeypatch.setattr(
        "frame.aggregate.calc_injected_t_significance_by_sqrt_q0_continuous",
        calculation,
    )
    expected = detector_efficiency_tanh(pd.DataFrame([[2.0, 1.0]])).item()
    np.testing.assert_allclose(
        ResultAggregator(tmp_path).all_injected_significances, [1.0, expected, expected]
    )
    assert len(calls) == 2
