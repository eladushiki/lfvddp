from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from data_tools.data_utils import DataSet
from neural_networks.differentiating_model import DifferentiatingModel
from plot.plot_factory import PlotFactory
from plot.plot_utils import utils__project_prediction_values_sliced
from plot.plots import (
    _display_edges_by_observable,
    _evaluate_prediction_grid,
    _prediction_grid,
    _spanning_dataset_from_observable_values,
)
from plot.plotting_config import PlotInstructions
from test.environment import ConfigType
from train.model_trainer import TrainLauncher


def test_prediction_grid_matches_legacy_projection_bitwise():
    display_edges_by_observable = {
        "x": np.array([-1.0, 0.0, 1.0]),
        "y": np.array([-2.0, -1.0, 0.0, 1.0]),
        "z": np.array([0.0, 1.0, 2.0]),
    }
    selected_observables = ["z", "x"]
    configured_observables = ["x", "y", "z"]
    prediction_grid = _prediction_grid(
        display_edges_by_observable=display_edges_by_observable,
        selected_observables=selected_observables,
        configured_observables=configured_observables,
        nuisance_spec=None,
        continuous_axis_points=4,
        chunk_size=7,
    )
    spanning_dataset = _spanning_dataset_from_observable_values(
        values_by_observable=dict(
            zip(configured_observables, prediction_grid.axes)
        ),
        observable_names=configured_observables,
    )

    def signal_plus(data: DataSet) -> np.ndarray:
        x, y, z = data.events.T
        return np.exp(x + y / 10.0 - z / 100.0)

    def signal_minus(data: DataSet) -> np.ndarray:
        x, y, z = data.events.T
        return np.sin(x) + np.cos(y) + z

    def numerator_theta(data: DataSet) -> np.ndarray:
        x, y, z = data.events.T
        return x * y - z

    def denominator_theta(data: DataSet) -> np.ndarray:
        x, y, z = data.events.T
        return x - y * z

    coordinates, projected_predictions = _evaluate_prediction_grid(
        prediction_grid,
        signal_plus_prediction_function=signal_plus,
        signal_minus_prediction_function=signal_minus,
        numerator_theta_prediction_function=numerator_theta,
        denominator_theta_prediction_function=denominator_theta,
    )
    legacy_coordinates, legacy_signal_plus = utils__project_prediction_values_sliced(
        values=signal_plus(spanning_dataset),
        spanning_dataset=spanning_dataset,
        along_observables=selected_observables,
    )

    np.testing.assert_array_equal(coordinates, legacy_coordinates)
    np.testing.assert_array_equal(
        projected_predictions.signal_plus, legacy_signal_plus
    )
    for prediction_function, projected_prediction in (
        (signal_minus, projected_predictions.signal_minus),
        (numerator_theta, projected_predictions.numerator_theta),
        (denominator_theta, projected_predictions.denominator_theta),
    ):
        _, legacy_prediction = utils__project_prediction_values_sliced(
            values=prediction_function(spanning_dataset),
            spanning_dataset=spanning_dataset,
            along_observables=selected_observables,
        )
        np.testing.assert_array_equal(projected_prediction, legacy_prediction)


@pytest.mark.parametrize(
    "function_execution_context",
    [
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
            id="2d-binned-nuisance",
        )
    ],
    indirect=True,
)
def test_prediction_grid_matches_full_model_evaluation_bitwise(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )
    numerator_model = differentiating_model_factory(
        function_execution_context,
        detector_effect,
        is_numerator=True,
        name="numerator_prediction_grid",
    )
    denominator_model = differentiating_model_factory(
        function_execution_context,
        detector_effect,
        is_numerator=False,
        name="denominator_prediction_grid",
    )
    numerator_model._prepare_training_data(detected_batch)
    denominator_model._prepare_training_data(detected_batch)

    configured_observables = list(detector_effect.observable_names)
    display_edges_by_observable = _display_edges_by_observable(
        datasets=[detected_batch.unified_data],
        observable_names=configured_observables,
        number_of_bins=(
            function_execution_context.config.plot__prediction_process_number_of_bins
        ),
    )
    prediction_grid = _prediction_grid(
        display_edges_by_observable=display_edges_by_observable,
        selected_observables=configured_observables,
        configured_observables=configured_observables,
        nuisance_spec=function_execution_context.config.train__function_space_config.nuisance,
        continuous_axis_points=(
            function_execution_context.config
            .plot__prediction_process_continuous_axis_points
        ),
        chunk_size=function_execution_context.config.plot__prediction_process_chunk_size,
    )
    spanning_dataset = _spanning_dataset_from_observable_values(
        values_by_observable=dict(
            zip(configured_observables, prediction_grid.axes)
        ),
        observable_names=configured_observables,
    )

    _, streamed_predictions = _evaluate_prediction_grid(
        prediction_grid,
        signal_plus_prediction_function=numerator_model.predict,
        signal_minus_prediction_function=numerator_model.predict_secondary,
        numerator_theta_prediction_function=numerator_model.predict_theta,
        denominator_theta_prediction_function=denominator_model.predict_theta,
    )

    np.testing.assert_array_equal(
        streamed_predictions.signal_plus,
        numerator_model.predict(spanning_dataset).reshape(-1),
    )
    np.testing.assert_array_equal(
        streamed_predictions.signal_minus,
        numerator_model.predict_secondary(spanning_dataset).reshape(-1),
    )
    np.testing.assert_array_equal(
        streamed_predictions.numerator_theta,
        numerator_model.predict_theta(spanning_dataset).reshape(-1),
    )
    np.testing.assert_array_equal(
        streamed_predictions.denominator_theta,
        denominator_model.predict_theta(spanning_dataset).reshape(-1),
    )

    projected_grid = _prediction_grid(
        display_edges_by_observable=display_edges_by_observable,
        selected_observables=[configured_observables[0]],
        configured_observables=configured_observables,
        nuisance_spec=function_execution_context.config.train__function_space_config.nuisance,
        continuous_axis_points=(
            function_execution_context.config
            .plot__prediction_process_continuous_axis_points
        ),
        chunk_size=10_000,
    )
    projected_spanning_dataset = _spanning_dataset_from_observable_values(
        values_by_observable=dict(
            zip(configured_observables, projected_grid.axes)
        ),
        observable_names=configured_observables,
    )
    coordinates, streamed_predictions = _evaluate_prediction_grid(
        projected_grid,
        signal_plus_prediction_function=numerator_model.predict,
        signal_minus_prediction_function=numerator_model.predict_secondary,
        numerator_theta_prediction_function=numerator_model.predict_theta,
        denominator_theta_prediction_function=denominator_model.predict_theta,
    )
    for prediction_function, streamed_prediction in (
        (numerator_model.predict, streamed_predictions.signal_plus),
        (numerator_model.predict_secondary, streamed_predictions.signal_minus),
        (numerator_model.predict_theta, streamed_predictions.numerator_theta),
        (denominator_model.predict_theta, streamed_predictions.denominator_theta),
    ):
        legacy_coordinates, legacy_prediction = (
            utils__project_prediction_values_sliced(
                values=prediction_function(projected_spanning_dataset),
                spanning_dataset=projected_spanning_dataset,
                along_observables=[configured_observables[0]],
            )
        )
        np.testing.assert_array_equal(coordinates, legacy_coordinates)
        np.testing.assert_array_equal(streamed_prediction, legacy_prediction)


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
            assert len(x_values) == (
                function_execution_context.config
                .plot__prediction_process_continuous_axis_points
            )
            assert (x_values[1:] > x_values[:-1]).all()
    plt.close(figure)
