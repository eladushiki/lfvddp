from pathlib import Path
import pytest
from test.environment import ConfigType
from data_tools.data_utils import DataSet


@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/configs/dataset/cms_open_dataset_json.json"),
    }, {
        ConfigType.DATASET.value: Path("test/configs/dataset/cms_open_dataset_txt.json"),
    }],
    indirect=True,
)
@pytest.mark.xfail  # Currently importing online root file from pytest context fails
def test_input_modes(
    data_generation,
):
    ds, _ = data_generation[DataSet.DataSetCategory.UNDEFINED]

    assert ds.n_samples > 5
