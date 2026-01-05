from pathlib import Path
import pytest
import numpy as np
from test.environment import ConfigType
from train.single_train import follow_instructions_for_t


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
    A, A_params = data_generation["A"]
    B, B_params = data_generation["B"]

    affected_A = detector_effect.affect_and_compensate(A, A_params, True)
    affected_B = detector_effect.affect_and_compensate(B, B_params, True)

    reference_dataset = affected_A + affected_B

    _, t_a_loss = follow_instructions_for_t(
        function_execution_context,
        affected_A,
        reference_dataset,
        detector_effect=detector_effect,
        name="test_model",
    )

    # Train should not yet converge but a value should be given
    assert t_a_loss != 0


@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_1D_detector_config.json"),
        ConfigType.TRAIN.value: Path("test/configs/train/long_1D_train_config_without_nuisance.json"),
    },
    {
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_1D_detector_config.json"),
        ConfigType.TRAIN.value: Path("test/configs/train/long_1D_train_config_with_nuisance.json"),
    }],
    indirect=True,
)
@pytest.mark.long
def test_convergence(
    function_execution_context,
    data_generation,
    detector_effect,
):
    A, A_params = data_generation["A"]
    B, B_params = data_generation["B"]

    affected_A = detector_effect.affect_and_compensate(A, A_params, True)
    affected_B = detector_effect.affect_and_compensate(B, B_params, True)

    reference_dataset = affected_A + affected_B

    model_a, t_a = follow_instructions_for_t(
        function_execution_context,
        affected_A,
        reference_dataset,
        detector_effect=detector_effect,
        name="test_model_A",
    )
    model_b, t_b = follow_instructions_for_t(
        function_execution_context,
        affected_B,
        reference_dataset,
        detector_effect=detector_effect,
        name="test_model_B",
    )

    # Load weights from both models
    weights_a = list(model_a.state_dict().values())
    weights_b = list(model_b.state_dict().values())

    # Verify weights are different
    for i, (w_a, w_b) in enumerate(zip(weights_a, weights_b)):
        assert not np.allclose(w_a.cpu().detach().numpy(), w_b.cpu().detach().numpy()), f"Weight matrix {i} should be different between models"

    assert t_a + t_b > 0
