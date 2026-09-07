from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from neural_networks.differentiating_model import DifferentiatingModel
from plot.plot_factory import PlotFactory
from plot.plotting_config import PlotInstructions
from plot.plots import _CONTINUOUS_PREDICTION_AXIS_POINTS
from test.environment import ConfigType
from train.model_trainer import TrainLauncher


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
