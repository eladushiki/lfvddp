from types import SimpleNamespace

import pytest

from data_tools.detector.detector_config import DetectorConfig
import plot.plots as plots
from plot.plot_factory import PlotFactory
from plot.plotting_config import PlotInstructions, PlotScope, PlottingConfig


def _plotting_config() -> PlottingConfig:
    return PlottingConfig(
        plot__target_run_parent_directory="",
        plot__pyplot_styling={"rcParams": {}, "style.use": "default"},
        plot__figure_styling={"patch_set_facecolor": "white"},
        plot__figure_size=(10, 9),
        plot__plot_specifications=[],
    )


class _PlottingDetectorConfig(PlottingConfig, DetectorConfig):
    def __init__(self, number_of_dimensions: int):
        PlottingConfig.__init__(self, **vars(_plotting_config()))
        self.detector__detect_observable_names = [
            f"observable_{index}" for index in range(number_of_dimensions)
        ]


def _plot_factory(number_of_dimensions: int) -> PlotFactory:
    context = SimpleNamespace(
        config=_PlottingDetectorConfig(number_of_dimensions)
    )
    return PlotFactory(context=context)


@pytest.mark.parametrize(
    ("number_of_dimensions", "function_name"),
    (
        (1, "plot_prediction_process_1d"),
        (2, "plot_prediction_process_2d"),
        (3, "plot_prediction_process_2d"),
    ),
)
def test_getitem_infers_prediction_process_dimension(
    number_of_dimensions, function_name
):
    plot_factory = _plot_factory(number_of_dimensions)

    assert plot_factory["plot_prediction_process"] is getattr(
        plots, function_name
    )


def test_generate_plot_uses_inferred_dimension_and_forwards_instructions(
    monkeypatch,
):
    plot_factory = _plot_factory(number_of_dimensions=1)
    received = {}

    def example_plot_1d(context, *, title):
        received.update(context=context, title=title)
        return "generated figure"

    monkeypatch.setattr(plots, "example_plot_1d", example_plot_1d, raising=False)
    instructions = PlotInstructions(
        name="example_plot", instructions={"title": "Example"}
    )

    result = plot_factory.generate_plot(instructions)

    assert result == "generated figure"
    assert received == {"context": plot_factory._context, "title": "Example"}


def test_getitem_rejects_dimension_inference_without_observables():
    plot_factory = _plot_factory(number_of_dimensions=0)

    with pytest.raises(ValueError, match="without configured detector observables"):
        plot_factory["missing_plot"]


def test_getitem_reports_missing_inferred_variant():
    plot_factory = _plot_factory(number_of_dimensions=2)

    with pytest.raises(
        KeyError,
        match="Expected function 'missing_plot_2d'",
    ):
        plot_factory["missing_plot"]


def test_exact_name_lookup_remains_available_without_detector_config():
    config = _plotting_config()
    plot_factory = PlotFactory(context=SimpleNamespace(config=config))

    assert plot_factory["t_train_percentile_progression_plot"] is (
        plots.t_train_percentile_progression_plot
    )


def test_missing_exact_name_requires_detector_config_for_inference():
    config = _plotting_config()
    plot_factory = PlotFactory(context=SimpleNamespace(config=config))

    with pytest.raises(KeyError, match="without a DetectorConfig"):
        plot_factory["missing_plot"]


def test_scope_selects_plots_declared_for_single_submission():
    config = _plotting_config()
    config.plot__plot_specifications = [
        {"name": "t_distribution_plot", "instructions": {}},
        {"name": "performance_plot", "instructions": {}},
    ]
    plot_factory = PlotFactory(context=SimpleNamespace(config=config))

    assert [
        instructions.name
        for instructions in plot_factory.plot_instructions_for_scope(
            PlotScope.SINGLE_SUBMISSION
        )
    ] == ["t_distribution_plot"]


def test_multi_run_scope_supplies_discovered_context_directories(monkeypatch, tmp_path):
    config = _plotting_config()
    config.plot__plot_specifications = [
        {"name": "performance_plot", "instructions": {}}
    ]
    plot_factory = PlotFactory(context=SimpleNamespace(config=config))
    background_directory = tmp_path / "background"
    monkeypatch.setattr(
        "plot.plot_factory.utils__discover_background_only_parent_directory",
        lambda _: background_directory,
    )

    [instructions] = plot_factory.plot_instructions_for_scope(
        PlotScope.MULTI_RUN, performance_directory=str(tmp_path)
    )

    assert instructions.instructions == {
        "background_only_t_values_parent_directory": str(background_directory),
        "signal_t_values_parent_directory": str(tmp_path),
    }
