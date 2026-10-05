from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from plot import plot_utils


class _FakeAggregator:
    def __init__(self, parent_directory: Path):
        self.parent_directory = parent_directory
        self.all_t_values = np.asarray([2.0, 3.0])

    @property
    def all_injected_significances(self):
        raise AssertionError(
            "Loaded-dataset performance curves must not calculate analytic "
            "injected significances."
        )


def _context_with_source(source_type: str, signal_events: int = 7):
    parameters = SimpleNamespace(
        dataset__background_source_type=source_type,
        dataset__mean_number_of_signal_events=signal_events,
    )
    return SimpleNamespace(
        config=SimpleNamespace(dataset_parameters=[parameters]),
        signal_parameters=parameters,
    )


def test_loaded_performance_curve_uses_signal_events_axis(monkeypatch, tmp_path):
    context = _context_with_source("loaded", signal_events=25)
    context_path = tmp_path / "signal" / "context.json"

    def fail_analytic_significance(**_arguments):
        raise AssertionError("Loaded datasets do not have an analytic background PDF.")

    monkeypatch.setattr(plot_utils, "ResultAggregator", _FakeAggregator)
    monkeypatch.setattr(
        plot_utils,
        "utils__get_signal_dataset_parameters",
        lambda signal_context: signal_context.signal_parameters,
    )
    monkeypatch.setattr(
        plot_utils,
        "calc_injected_t_significance_by_sqrt_q0_continuous",
        fail_analytic_significance,
    )

    curve = plot_utils.utils__calculate_performance_curve(
        [(context, context_path)],
        background_t_dist=np.asarray([0.5, 1.0, 1.5]),
    )

    np.testing.assert_array_equal(curve.x_values, [25])
    np.testing.assert_array_equal(curve.x_errors, [0.0])
    assert curve.x_label == "mean injected signal events"
    assert curve.show_reference_diagonal is False
    assert curve.connect_points is False


def test_generated_performance_curve_keeps_analytic_significance(
    monkeypatch,
    tmp_path,
):
    context = _context_with_source("generated", signal_events=25)
    context.signal_parameters.dataset_generated__background_pdf = lambda x: 1.0
    context.signal_parameters.dataset_generated__signal_pdf = lambda x: 1.0
    context.signal_parameters.dataset__mean_number_of_background_events = 100
    context.signal_parameters.dataset_generated__integration_upper_limits = 10.0
    context_path = tmp_path / "signal" / "context.json"

    class GeneratedAggregator(_FakeAggregator):
        @property
        def all_injected_significances(self):
            return np.asarray([3.0, 4.0, 5.0])

    monkeypatch.setattr(plot_utils, "ResultAggregator", GeneratedAggregator)
    monkeypatch.setattr(
        plot_utils,
        "utils__get_signal_dataset_parameters",
        lambda signal_context: signal_context.signal_parameters,
    )
    monkeypatch.setattr(
        plot_utils,
        "calc_injected_t_significance_by_sqrt_q0_continuous",
        lambda **_arguments: 4.5,
    )

    curve = plot_utils.utils__calculate_performance_curve(
        [(context, context_path)],
        background_t_dist=np.asarray([0.5, 1.0, 1.5]),
    )

    np.testing.assert_array_equal(curve.x_values, [4.5])
    np.testing.assert_allclose(curve.x_errors, [np.std([3.0, 4.0, 5.0])])
    assert curve.x_label == r"injected $\sqrt{q_0}$"
    assert curve.show_reference_diagonal is True
    assert curve.connect_points is True


def test_performance_curve_coalesces_duplicate_signal_strengths(
    monkeypatch,
    tmp_path,
):
    context = _context_with_source("loaded", signal_events=25)
    first_path = tmp_path / "first" / "context.json"
    second_path = tmp_path / "second" / "context.json"

    class DuplicateAggregator(_FakeAggregator):
        def __init__(self, parent_directory: Path):
            self.parent_directory = parent_directory
            values_by_directory = {
                "first": np.asarray([2.0, 3.0]),
                "second": np.asarray([4.0, 5.0]),
            }
            self.all_t_values = values_by_directory[parent_directory.name]

    monkeypatch.setattr(plot_utils, "ResultAggregator", DuplicateAggregator)
    monkeypatch.setattr(
        plot_utils,
        "utils__get_signal_dataset_parameters",
        lambda signal_context: signal_context.signal_parameters,
    )

    background_t_dist = np.asarray([0.5, 1.0, 1.5])
    curve = plot_utils.utils__calculate_performance_curve(
        [(context, first_path), (context, second_path)],
        background_t_dist=background_t_dist,
    )

    np.testing.assert_array_equal(curve.x_values, [25])
    expected_significance = plot_utils.calc_mean_t_significance_relative_to_background(
        background_t_dist,
        np.asarray([2.0, 3.0, 4.0, 5.0]),
    )
    np.testing.assert_allclose(curve.observed_significances, [expected_significance])


def test_performance_curve_rejects_mixed_generated_and_loaded(tmp_path):
    loaded_context = _context_with_source("loaded")
    generated_context = _context_with_source("generated")

    with pytest.raises(
        ValueError,
        match="Mixed generated and loaded datasets",
    ):
        plot_utils.utils__calculate_performance_curve(
            [
                (loaded_context, tmp_path / "loaded" / "context.json"),
                (generated_context, tmp_path / "generated" / "context.json"),
            ],
            background_t_dist=np.asarray([0.5, 1.0, 1.5]),
        )
