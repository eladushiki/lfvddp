from pathlib import Path

import numpy as np
import pytest
import torch

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from frame.file_system.training_history import HistoryKeys
from neural_networks.differentiating_model import (
    _calculate_loss_weights,
    _expand_masked_predictions,
)
from test.environment import ConfigType
from train.checkpoints import _torch_load, checkpoint_filename
from train.model_trainer import SequentialTrainLauncher
from train.tensorboard_clutch import log_t_history


def _train_numerator(function_execution_context, data_batch, detector_effect, name):
    train_launcher = SequentialTrainLauncher(
        function_execution_context, detector_effect
    )
    train_idx = train_launcher.add_training(
        data_batch=data_batch,
        detector_effect=detector_effect,
        is_numerator=True,
        name=name,
    )
    train_launcher.execute_trainings()
    return train_launcher.get_train_result(train_idx)


def _load_checkpoint_state_dict(function_execution_context, model_name):
    checkpoint = _torch_load(
        function_execution_context.training_outcomes_dir
        / checkpoint_filename(model_name)
    )
    return checkpoint["model_state_dict"]


def test_loss_weights_follow_weight_order():
    category_sizes = {
        DataSet.DataSetCategory.A_SR: 1,
        DataSet.DataSetCategory.A_CR: 2,
        DataSet.DataSetCategory.B_SR: 3,
        DataSet.DataSetCategory.B_CR: 4,
    }
    datasets = []
    for category, size in reversed(category_sizes.items()):
        dataset = DataSet(
            data=np.full((size, 2), category.value),
            observable_names=["observable_1", "observable_2"],
            category=category,
        )
        datasets.append((dataset, None))

    data_batch = DataBatch(datasets)

    np.testing.assert_array_equal(
        data_batch.unified_data.events[:, 0],
        [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    )
    np.testing.assert_array_equal(
        _calculate_loss_weights(data_batch),
        np.ones((10, 1)),
    )


def test_masked_predictions_are_restored_to_batch_positions():
    predictions = torch.tensor([[10.0], [30.0]], requires_grad=True)
    sr_mask = torch.tensor([True, False, True, False])

    expanded = _expand_masked_predictions(predictions, sr_mask)

    torch.testing.assert_close(expanded, torch.tensor([10.0, 0.0, 30.0, 0.0]))
    expanded.sum().backward()
    torch.testing.assert_close(predictions.grad, torch.ones_like(predictions))


class _TensorboardRecorder:
    def __init__(self):
        self.scalars = []
        self.histograms = []

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))

    def add_histogram(self, tag, values, global_step):
        self.histograms.append((tag, values, global_step))


@pytest.mark.parametrize(
    "function_execution_context",
    [{  # basic process
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_1D_detector_config.json"),
        ConfigType.TRAIN.value: Path("test/configs/train/short_1D_train_config_with_nuisance.json"),
    }, {  # basic without nuisance
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_1D_detector_config.json"),
        ConfigType.TRAIN.value: Path("test/configs/train/short_1D_train_config_without_nuisance.json"),
    }, {
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_1D_detector_config.json"),
        ConfigType.TRAIN.value: Path("test/configs/train/short_1D_train_config_without_nuisance_like_nplm.json"),
    }],
    indirect=True,
)
def test_learning(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(
        data_generation.get_batch()
    )
    t_a_loss = _train_numerator(
        function_execution_context,
        detected_batch,
        detector_effect,
        "test_model",
    )

    # Train should not yet converge but a value should be given
    assert t_a_loss != 0


@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path(
            "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
        ),
        ConfigType.DETECTOR.value: Path(
            "test/configs/detector/basic_1D_detector_config.json"
        ),
        ConfigType.TRAIN.value: Path(
            "test/configs/train/profile_1D_train_config_with_nuisance.json"
        ),
    }],
    indirect=True,
)
def test_training_profile_is_saved(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    _train_numerator(
        function_execution_context,
        detected_batch,
        detector_effect,
        "profiled_model",
    )

    profile_stem = "profiled_model.1D.profile"
    trace_path = (
        function_execution_context.training_outcomes_dir
        / f"{profile_stem}.trace.json"
    )
    summary_path = (
        function_execution_context.training_outcomes_dir / f"{profile_stem}.txt"
    )

    assert trace_path.is_file()
    summary = summary_path.read_text()
    assert "observables: 1" in summary
    assert "training/forward_and_loss" in summary
    assert "training/nuisance_eta" in summary
    assert "training/backward" in summary


def test_t_history_is_logged_to_tensorboard():
    recorder = _TensorboardRecorder()
    history = {
        HistoryKeys.EPOCH.value: [7, 9],
        HistoryKeys.NUMERATOR.value: [1.5, 1.25],
        HistoryKeys.DENOMINATOR.value: [2.0, 1.75],
        HistoryKeys.T.value: [1.0, 1.0],
    }

    log_t_history(recorder, "A", history)

    assert recorder.scalars == [
        ("A/numerator", 1.5, 7),
        ("A/numerator", 1.25, 9),
        ("A/denominator", 2.0, 7),
        ("A/denominator", 1.75, 9),
        ("A/t", 1.0, 7),
        ("A/t", 1.0, 9),
    ]
    assert recorder.histograms == []


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/long_1D_train_config_without_nuisance.json"
            ),
        },
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/long_1D_train_config_with_nuisance.json"
            ),
        },
    ],
    indirect=True,
)
@pytest.mark.long
def test_convergence(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(
        data_generation.get_batch()
    )
    t_a = _train_numerator(
        function_execution_context,
        detected_batch,
        detector_effect,
        "test_model_A",
    )
    
    assert t_a > 0
