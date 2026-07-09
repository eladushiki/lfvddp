from pathlib import Path
import pytest
import torch
from neural_networks.differentiating_model import DifferentiatingModel
from test.environment import ConfigType
from train.checkpoints import _torch_load, checkpoint_filename
from train.model_trainer import SequentialTrainLauncher


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
    detected_batch = detector_effect.affect_and_compensate_batch(
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
    [
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_1D_train_config_with_nuisance.json"
            ),
        }
    ],
    indirect=True,
)
def test_differentiating_model_logs_parameters_to_tensorboard(
    function_execution_context,
    detector_effect,
):
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="test_model",
    )
    recorder = _TensorboardRecorder()
    model._tensorboard_writer = recorder

    model._log_epoch(7, torch.tensor(1.5))

    assert [tag for tag, _, _ in recorder.scalars] == ["loss"]
    assert recorder.histograms == []

    model._log_epoch(9, torch.tensor(1.25))

    scalar_tags = [tag for tag, _, _ in recorder.scalars]
    histogram_tags = [tag for tag, _, _ in recorder.histograms]

    assert scalar_tags == ["loss", "loss"]
    assert any(tag.startswith("parameters/f_network/") for tag in histogram_tags)
    assert any(tag.startswith("parameters/g_network/") for tag in histogram_tags)
    assert any(tag.startswith("parameters/eta/nuisance_") for tag in histogram_tags)
    assert all(global_step == 9 for _, _, global_step in recorder.histograms)


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
    detected_batch = detector_effect.affect_and_compensate_batch(
        data_generation.get_batch()
    )
    t_a = _train_numerator(
        function_execution_context,
        detected_batch,
        detector_effect,
        "test_model_A",
    )
    detected_batch.swap_ab()
    t_b = _train_numerator(
        function_execution_context,
        detected_batch,
        detector_effect,
        "test_model_B",
    )

    weights_a = list(
        _load_checkpoint_state_dict(function_execution_context, "test_model_A").values()
    )
    weights_b = list(
        _load_checkpoint_state_dict(function_execution_context, "test_model_B").values()
    )

    # Verify weights are different
    assert any(
        not torch.allclose(w_a.cpu(), w_b.cpu())
        for w_a, w_b in zip(weights_a, weights_b)
    )

    assert t_a + t_b > 0
