from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from neural_networks.differentiating_model import DifferentiatingModel
from plot.plots import plot_prediction_process_1d
from test.environment import ConfigType
from train.model_trainer import TrainLauncher


@pytest.mark.parametrize(
    "function_execution_context",
    [
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
        }
    ],
    indirect=True,
)
def test_prediction_process_1d_supports_neural_nuisance(
    function_execution_context,
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

    figure = plot_prediction_process_1d(
        context=function_execution_context,
        numerator_training=prepared_training(is_numerator=True),
        denominator_training=prepared_training(is_numerator=False),
    )

    assert len(figure.axes) == 4
    assert all(
        axis.lines or axis.patches or axis.collections for axis in figure.axes
    )
    plt.close(figure)
