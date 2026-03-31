import pytest
from pathlib import Path
from test.environment import ConfigType


@pytest.mark.parametrize(
        "function_execution_context",
        [{
            ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
        }],
        indirect=True,
)
def test_illegal_data_request(
        data_generation,
):
    with pytest.raises(KeyError):
        _, _ = data_generation["non_existing_dataset"]

@pytest.mark.parametrize(
        "function_execution_context",
        [{
            ConfigType.DATASET.value: Path("test/data_generation/configs/dataset/small_exact_sized_loaded_dataset_config.json"),
        }],
        indirect=True,
)
@pytest.mark.xfail  # Currently online data loading from pytest context causes trouble
def test_resampling_exhaustion(
        data_generation,
):
    _, _ = data_generation["A"]

    # Draws more data than there is left, due to resampling without replacement
    with pytest.raises(ValueError):
        _, _ = data_generation["A"]
