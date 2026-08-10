from pathlib import Path
import numpy as np
import pytest

from test.environment import ConfigType
from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetConfig

@pytest.mark.parametrize(
    "function_execution_context",
    [{
        ConfigType.DATASET.value: Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json"),
    }],
    indirect=True,
)
def test_dataset_normalization(
    function_execution_context,
    data_generation,
):
    A, _ = data_generation[DataSet.DataSetCategory.A_SR]
    B, _ = data_generation[DataSet.DataSetCategory.B_SR]

    normalized_A, norm_factor_A = A.get_normalized()
    normalized_B, norm_factor_B = B.get_normalized()

    np.testing.assert_allclose(np.max(normalized_A.events), 1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.max(normalized_B.events), 1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.min(normalized_A.events), -1, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.min(normalized_B.events), -1, rtol=1e-9, atol=1e-9)

    np.testing.assert_array_almost_equal((normalized_A * norm_factor_A).events, A.events)
    np.testing.assert_array_almost_equal((normalized_B * norm_factor_B).events, B.events)


def test_dataset_config_owns_signal_configuration_fields():
    assert DatasetConfig.SIGNAL_EVENT_CONFIGURATION_FIELDS == {
        "dataset__mean_number_of_signal_events",
    }
    assert DatasetConfig.SIGNAL_CONFIGURATION_FIELDS == {
        "dataset__mean_number_of_signal_events",
        "dataset__signal_generator",
    }
