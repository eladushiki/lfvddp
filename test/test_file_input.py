from pathlib import Path

import pytest

from data_tools.data_utils import DataSet
from test.environment import ConfigType


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/cms_open_dataset_root.json"
            ),
        }
    ],
    indirect=True,
)
@pytest.mark.integration
@pytest.mark.remote
def test_remote_root_input_honors_event_limit(
    function_execution_context,
):
    parameters = function_execution_context.config.get_parameters(
        DataSet.DataSetCategory.A_SR
    )
    dataset, _ = parameters.dataset__data

    assert dataset.n_samples == parameters.dataset_loaded__event_amount_load_limit
    assert dataset.observable_names == ["run", "event"]
