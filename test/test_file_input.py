from pathlib import Path
import pytest
from test.environment import ConfigType


@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/configs/dataset/cms_open_dataset_json.json"),
    }, {
        ConfigType.DATASET.value: Path("test/configs/dataset/cms_open_dataset_json_txt.json"),
    }],
    indirect=True,
)
def test_input_modes(
    data_generation,
):
    ds, _ = data_generation["cms_sample"]

    assert ds.n_samples > 5
