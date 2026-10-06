from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from data_tools.data_utils import DataSet
from data_tools.detector.detector_config import DetectorConfig
from data_tools.histogram_binning import display_edges_by_observable
from frame.file_system.data_samples import load_data_samples, save_data_samples
from plot import plots
from plot import plot_utils
from plot.plotting_config import PlottingConfig
from train.single_train import save_training_data_samples


class _PerformanceConfig(PlottingConfig, DetectorConfig):
    def __init__(self, dataset_parameters):
        PlottingConfig.__init__(self, plot__plot_specifications=[])
        DetectorConfig.__init__(self, detector__detect_observable_names=["x"])
        self.dataset_parameters = dataset_parameters
        self.plot__prediction_process_number_of_bins = 2
        self.train__function_space_config = SimpleNamespace(nuisance=None)


class _FakeAggregator:
    def __init__(self, parent_directory: Path):
        self.parent_directory = parent_directory
        self.all_t_values = np.asarray([20.0, 30.0, 100.0])

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
        dataset__mean_number_of_background_events=100,
    )
    return SimpleNamespace(
        config=_PerformanceConfig(dataset_parameters=[parameters]),
        signal_parameters=parameters,
    )


def _save_sample(path, background, signal, number_of_bins=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    background_data = DataSet(np.asarray(background), ["x"])
    signal_data = DataSet(np.asarray(signal), ["x"])
    save_data_samples(
        path.parent,
        background_data,
        signal_data,
        ["x"],
        display_edges_by_observable(
            [background_data, signal_data], ["x"], number_of_bins
        ),
    )


def test_loaded_performance_curve_uses_binned_evident_significance(
    monkeypatch,
    tmp_path,
):
    context = _context_with_source("loaded", signal_events=25)
    context.signal_parameters.dataset__data = (
        DataSet(np.asarray([[0.25], [0.75]]), ["x"]),
        DataSet(np.asarray([[0.75], [0.75]]), ["x"]),
    )
    context_path = tmp_path / "signal" / "context.json"
    _save_sample(context_path, [[0.25], [0.75]], [[0.75], [0.75]])
    context.signal_parameters.dataset__data = None

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
        background_t_dist=np.arange(100, dtype=float),
    )

    expected_x_value = plot_utils.calc_injected_t_significance_by_sqrt_q0_binned(
        background_bin_counts=np.asarray([50.0, 50.0]),
        signal_bin_counts=np.asarray([0.0, 25.0]),
    )
    np.testing.assert_allclose(curve.x_values, [expected_x_value])
    np.testing.assert_array_equal(curve.x_errors, [0.0])
    assert curve.observed_significances[0] == pytest.approx(
        plot_utils.calc_t_significance_relative_to_background(
            30.0,
            np.arange(100, dtype=float),
        )
    )
    assert curve.gaussian_fit_significances[0] == pytest.approx(
        plot_utils.calc_t_significance_by_gaussian_fit_percentile(
            background_only_distribution=np.arange(100, dtype=float),
            t_value=30.0,
        )
    )
    assert curve.x_label == r"evident injected $\sqrt{q_0}$"
    assert curve.show_reference_diagonal is False
    assert curve.connect_points is False


def test_loaded_significance_uses_training_sample(monkeypatch, tmp_path):
    context = _context_with_source("loaded", signal_events=25)
    context.signal_parameters.category = DataSet.DataSetCategory.B_SR
    context.signal_parameters.dataset__has_signal = True
    context.config.dataset__has_signal = True
    context.config.get_parameters = lambda category: context.signal_parameters
    context.unique_out_dir = tmp_path / "signal"
    context.unique_out_dir.mkdir()
    context_path = context.unique_out_dir / "context.json"

    class SampledGeneration:
        def sampled_components(self, category):
            assert category == DataSet.DataSetCategory.B_SR
            return (
                DataSet(np.asarray([[0.25], [0.75]]), ["x"]),
                DataSet(np.asarray([[0.75], [0.75]]), ["x"]),
            )

    detected_batch = SimpleNamespace(
        unified_data=DataSet(np.asarray([[-1.0], [3.0]]), ["x"])
    )
    save_training_data_samples(context, SampledGeneration(), detected_batch)
    samples = load_data_samples(context.unique_out_dir)
    np.testing.assert_array_equal(
        samples.bin_edges_by_observable["x"], [-1.0, 1.0, 3.0]
    )
    context.signal_parameters.dataset__data = (
        DataSet(np.asarray([[0.25], [0.25]]), ["x"]),
        DataSet(np.asarray([[0.25], [0.25]]), ["x"]),
    )
    monkeypatch.setattr(plot_utils, "ResultAggregator", _FakeAggregator)
    monkeypatch.setattr(
        plot_utils,
        "utils__get_signal_dataset_parameters",
        lambda _context: context.signal_parameters,
    )

    curve = plot_utils.utils__calculate_performance_curve(
        [(context, context_path)], np.arange(100, dtype=float)
    )
    expected = plot_utils.calc_injected_t_significance_by_sqrt_q0_binned(
        np.asarray([100.0, 0.0]), np.asarray([25.0, 0.0])
    )
    np.testing.assert_allclose(curve.x_values, [expected])


def test_loaded_performance_curve_uses_prediction_plot_bins(monkeypatch, tmp_path):
    context = _context_with_source("loaded", signal_events=25)
    context.config.plot__prediction_process_number_of_bins = 1
    context.config.train__function_space_config = SimpleNamespace(
        nuisance=SimpleNamespace(
            family="bin_indicators",
            options={
                "minima": [0.0],
                "maxima": [1.0],
                "number_of_bins": [2],
            },
        )
    )
    context.signal_parameters.dataset__data = (
        DataSet(np.asarray([[0.25], [0.75]]), ["x"]),
        DataSet(np.asarray([[0.75], [0.75]]), ["x"]),
    )
    context_path = tmp_path / "signal" / "context.json"
    _save_sample(context_path, [[0.25], [0.75]], [[0.75], [0.75]], number_of_bins=1)

    monkeypatch.setattr(plot_utils, "ResultAggregator", _FakeAggregator)
    monkeypatch.setattr(
        plot_utils,
        "utils__get_signal_dataset_parameters",
        lambda signal_context: signal_context.signal_parameters,
    )

    curve = plot_utils.utils__calculate_performance_curve(
        [(context, context_path)],
        background_t_dist=np.arange(100, dtype=float),
    )

    expected_x_value = plot_utils.calc_injected_t_significance_by_sqrt_q0_binned(
        background_bin_counts=np.asarray([100.0]),
        signal_bin_counts=np.asarray([25.0]),
    )
    np.testing.assert_allclose(curve.x_values, [expected_x_value])


def test_loaded_performance_plot_keeps_gaussian_fit_dashed_curve(
    monkeypatch,
    tmp_path,
):
    context = SimpleNamespace(
        config=PlottingConfig(plot__plot_specifications=[]),
        run_hash=123,
    )
    background_context = SimpleNamespace(
        config=SimpleNamespace(
            dataset_parameters=[
                SimpleNamespace(dataset__has_signal=False),
            ],
        ),
    )
    signal_context = SimpleNamespace()
    curve = plot_utils._PerformanceCurve(
        x_values=np.asarray([10.0, 20.0, 30.0]),
        x_errors=np.asarray([0.0, 0.0, 0.0]),
        x_label="mean injected signal events",
        show_reference_diagonal=False,
        connect_points=False,
        observed_significances=np.asarray([1.0, 1.5, 2.0]),
        observed_significance_lower_bounds=np.asarray([0.8, 1.2, 1.7]),
        observed_significance_upper_bounds=np.asarray([1.2, 1.8, 2.3]),
        gaussian_fit_significances=np.asarray([0.9, 1.4, 1.9]),
    )

    monkeypatch.setattr(
        plots,
        "utils__discover_performance_contexts",
        lambda _directory: [(background_context, tmp_path / "bkg" / "context.json")],
    )
    monkeypatch.setattr(
        plots,
        "utils__warn_for_context_discrepancies",
        lambda *_, **__: None,
    )
    monkeypatch.setattr(
        plots,
        "utils__aggregate_context_t_values",
        lambda _contexts: np.asarray([0.0, 1.0, 2.0]),
    )
    monkeypatch.setattr(
        plots,
        "utils__group_signal_contexts",
        lambda _directory: [[(signal_context, tmp_path / "signal" / "context.json")]],
    )
    monkeypatch.setattr(
        plots,
        "utils__context_background_source_type",
        lambda _context: "loaded",
    )
    monkeypatch.setattr(
        plots,
        "utils__calculate_performance_curve",
        lambda _signal_group, _background_t_dist: curve,
    )
    monkeypatch.setattr(
        plots,
        "utils__performance_group_label",
        lambda _context: "signal",
    )

    figure = plots.performance_plot(
        context,
        background_only_t_values_parent_directory=str(tmp_path / "bkg"),
        signal_t_values_parent_directory=str(tmp_path / "signal"),
    )

    dashed_lines = [
        line for line in figure.axes[0].lines if line.get_linestyle() == "--"
    ]
    assert len(dashed_lines) == 1
    np.testing.assert_array_equal(dashed_lines[0].get_xdata(), curve.x_values)
    np.testing.assert_array_equal(
        dashed_lines[0].get_ydata(),
        curve.gaussian_fit_significances,
    )
    assert not any(
        line.get_linestyle() == "-"
        and np.array_equal(line.get_xdata(), curve.x_values)
        and np.array_equal(line.get_ydata(), curve.observed_significances)
        for line in figure.axes[0].lines
    )
    plt.close(figure)


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

    monkeypatch.setattr(plot_utils, "ResultAggregator", _FakeAggregator)
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
    np.testing.assert_allclose(curve.x_errors, [0.0])
    assert curve.x_label == r"injected $\sqrt{q_0}$"
    assert curve.show_reference_diagonal is True
    assert curve.connect_points is True


def test_performance_curve_coalesces_duplicate_signal_strengths(
    monkeypatch,
    tmp_path,
):
    context = _context_with_source("loaded", signal_events=25)
    context.signal_parameters.dataset__data = (
        DataSet(np.asarray([[0.25], [0.75]]), ["x"]),
        DataSet(np.asarray([[0.75], [0.75]]), ["x"]),
    )
    first_path = tmp_path / "first" / "context.json"
    second_path = tmp_path / "second" / "context.json"
    _save_sample(first_path, [[0.25], [0.75]], [[0.75], [0.75]])
    _save_sample(second_path, [[0.25], [0.75]], [[0.75], [0.75]])

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

    expected_x_value = plot_utils.calc_injected_t_significance_by_sqrt_q0_binned(
        background_bin_counts=np.asarray([50.0, 50.0]),
        signal_bin_counts=np.asarray([0.0, 25.0]),
    )
    np.testing.assert_allclose(curve.x_values, [expected_x_value])
    expected_significance = (
        plot_utils.calc_median_t_significance_relative_to_background(
            background_t_dist,
            np.asarray([2.0, 3.0, 4.0, 5.0]),
        )
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
