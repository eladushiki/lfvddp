from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from data_tools.data_utils import DataSet
from neural_networks.differentiating_model import DifferentiatingModel
from plot.plot_factory import PlotFactory
from plot.plotting_config import PlotInstructions, PlottingConfig
from plot.plots import _CONTINUOUS_PREDICTION_AXIS_POINTS
from test.environment import ConfigType
from train.model_trainer import TrainLauncher
from train.single_train import plot_training_prediction


@pytest.mark.parametrize(
    ("function_execution_context", "number_of_dimensions"),
    [
        pytest.param(
            {
                ConfigType.DATASET: Path(
                    "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
                ),
                ConfigType.DETECTOR: Path(
                    "test/configs/detector/basic_1D_detector_config.json"
                ),
                ConfigType.TRAIN: Path(
                    "test/configs/train/short_1D_train_config_with_neural_nuisance.json"
                ),
            },
            1,
            id="1d-neural-nuisance",
        ),
        pytest.param(
            {
                ConfigType.DATASET: Path(
                    "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
                ),
                ConfigType.DETECTOR: Path(
                    "test/configs/detector/basic_1D_detector_config.json"
                ),
                ConfigType.TRAIN: Path(
                    "test/configs/train/short_1D_train_config_without_nuisance.json"
                ),
            },
            1,
            id="1d-disabled-nuisance",
        ),
        pytest.param(
            {
                ConfigType.DATASET: Path(
                    "test/configs/dataset/disjoint_2D_generated_dataset_config.json"
                ),
                ConfigType.DETECTOR: Path(
                    "test/configs/detector/basic_2D_detector_config.json"
                ),
                ConfigType.TRAIN: Path(
                    "test/configs/train/short_2D_train_config_with_nuisance.json"
                ),
            },
            2,
            id="2d-binned-nuisance",
        ),
    ],
    indirect=["function_execution_context"],
)
def test_prediction_process_plot_generation(
    function_execution_context,
    number_of_dimensions,
    isolated_data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )

    def prepared_training(is_numerator: bool) -> TrainLauncher.Training:
        model = DifferentiatingModel(
            context=function_execution_context,
            detector_effect=detector_effect,
            is_numerator=is_numerator,
            name=f"neural_nuisance_{is_numerator}",
        )
        model._prepare_training_data(detected_batch)
        return TrainLauncher.Training(
            data_batch=detected_batch,
            detector_effect=detector_effect,
            is_numerator=is_numerator,
            model=model,
        )

    figure = PlotFactory(function_execution_context).generate_plot(
        PlotInstructions(
            name="plot_prediction_process",
            instructions={
                "numerator_training": prepared_training(is_numerator=True),
                "denominator_training": prepared_training(is_numerator=False),
            },
        )
    )

    assert len(figure.axes) == 4
    assert all(
        axis.lines or axis.patches or axis.collections for axis in figure.axes
    )
    if number_of_dimensions == 1:
        prediction_lines = [
            line
            for axis in figure.axes[2:]
            for line in axis.lines
            if "hypothesis" in line.get_label()
        ]
        assert prediction_lines
        for line in prediction_lines:
            x_values = line.get_xdata()
            assert len(x_values) == _CONTINUOUS_PREDICTION_AXIS_POINTS
            assert (x_values[1:] > x_values[:-1]).all()
    plt.close(figure)


@pytest.mark.parametrize(
    ("configured_title", "expected_title"),
    (
        (None, "sample prediction process"),
        ("Configured prediction", "Configured prediction"),
    ),
)
def test_training_prediction_plot_uses_configured_title(
    monkeypatch,
    tmp_path,
    configured_title,
    expected_title,
):
    received = {}

    class FakePlotFactory:
        def __init__(self, context):
            received["context"] = context

        def generate_plot(self, instructions):
            received["instructions"] = instructions
            return "figure"

    monkeypatch.setattr("plot.plot_factory.PlotFactory", FakePlotFactory)
    config = PlottingConfig(
        plot__plot_specifications=[],
        plot__prediction_process_title=configured_title,
    )
    context = SimpleNamespace(
        config=config,
        is_debug_mode=True,
        unique_out_dir=tmp_path,
        save_and_document_figure=lambda figure, path: received.update(
            figure=figure, path=path
        ),
    )
    training = SimpleNamespace(
        model=object(),
        data_batch=SimpleNamespace(
            parameters={DataSet.DataSetCategory.A_SR: SimpleNamespace(name="sample")}
        ),
    )

    plot_training_prediction(context, training, training)

    assert received["instructions"].instructions["title"] == expected_title
