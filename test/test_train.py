from pathlib import Path
import pytest
from test.environment import ConfigType
from train.single_train import follow_instructions_for_t


@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        ConfigType.DETECTOR.value: Path("test/configs/detector/basic_1D_detector_config.json"),
        ConfigType.TRAIN.value: Path("test/configs/train/short_1D_train_config.json"),
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

    t_a_loss = follow_instructions_for_t(
        function_execution_context,
        affected_A,
        reference_dataset,
        detector_effect=detector_effect,
        name="test_model",
    )

    assert t_a_loss > 0
