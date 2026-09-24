"""Physical-coordinate geometry must share the model input normalization."""

from pathlib import Path

import numpy as np
import pytest
import torch

from data_tools.data_utils import DataSet
from neural_networks.function_spaces import create_function_space
from test.environment import ConfigType


_DATASET_CONFIG = Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json")
_DETECTOR_CONFIG = Path("test/configs/detector/basic_1D_detector_config.json")


def _context_params(train_config: str):
    return {
        ConfigType.DATASET: _DATASET_CONFIG,
        ConfigType.DETECTOR: _DETECTOR_CONFIG,
        ConfigType.TRAIN: Path("test/configs/train") / train_config,
    }


def _raw_signal_events(data):
    return np.concatenate(
        (
            data.datasets[DataSet.DataSetCategory.A_SR].events,
            data.datasets[DataSet.DataSetCategory.B_SR].events,
        )
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(_context_params("cubic_bspline_binned.json"), id="cubic-bspline"),
        pytest.param(_context_params("orthogonal_legendre_binned.json"), id="legendre"),
        pytest.param(_context_params("orthogonal_chebyshev_binned.json"), id="chebyshev"),
        pytest.param(_context_params("fixed_sigmoid_binned.json"), id="fixed-sigmoid"),
        pytest.param(_context_params("gaussian_radial_basis_binned.json"), id="gaussian-radial-basis"),
    ],
    indirect=True,
)
def test_signal_geometry_from_config_is_evaluated_in_physical_coordinates(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    data = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = differentiating_model_factory(
        function_execution_context, detector_effect, name="physical_signal_geometry"
    )
    prepared = model._prepare_training_data(data)
    spec = function_execution_context.config.train__function_space_config.f
    raw_space = create_function_space(spec, dtype=torch.float64)

    assert model.signal_region_shift_network is not None
    torch.testing.assert_close(
        model.signal_region_shift_network.features(prepared.sr_events),
        raw_space.features(torch.tensor(_raw_signal_events(data), dtype=torch.float64)),
    )
    assert model.signal_region_shift_network.geometry == raw_space.geometry


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(_context_params("cubic_bspline_cubic_bspline.json"), id="cubic-bspline"),
        pytest.param(_context_params("cubic_bspline_fixed_sigmoid.json"), id="fixed-sigmoid"),
    ],
    indirect=True,
)
def test_nuisance_geometry_uses_the_same_physical_to_model_map(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    data = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = differentiating_model_factory(
        function_execution_context, detector_effect, name="physical_nuisance_geometry"
    )
    prepared = model._prepare_training_data(data)
    spec = function_execution_context.config.train__function_space_config.nuisance
    assert spec is not None
    assert model.nuisance_function_space is not None
    raw_space = create_function_space(spec, dtype=torch.float64)

    raw_sr = torch.tensor(_raw_signal_events(data), dtype=torch.float64)
    normalized_sr = prepared.nuisance_events[: raw_sr.shape[0]]
    torch.testing.assert_close(
        model.nuisance_function_space.features(normalized_sr), raw_space.features(raw_sr)
    )
    assert model.nuisance_function_space.geometry == raw_space.geometry


@pytest.mark.parametrize(
    "function_execution_context",
    [pytest.param(_context_params("bin_indicators_binned.json"), id="binned")],
    indirect=True,
)
def test_binned_signal_and_nuisance_share_the_same_normalized_basis(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    data = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = differentiating_model_factory(
        function_execution_context, detector_effect, name="physical_binned_geometry"
    )
    prepared = model._prepare_training_data(data)
    assert model.signal_region_shift_network is not None
    assert model.nuisance_function_space is not None
    raw_signal = torch.tensor(_raw_signal_events(data), dtype=torch.float64)
    spec = function_execution_context.config.train__function_space_config.f
    raw_space = create_function_space(spec, dtype=torch.float64)
    with torch.no_grad():
        values = torch.arange(1, raw_space.feature_count + 1, dtype=torch.float64)[:, None]
        raw_space.coefficients.copy_(values)
        model.signal_region_shift_network.coefficients.copy_(values)
        model.nuisance_function_space.coefficients.copy_(values)

    torch.testing.assert_close(
        model.signal_region_shift_network(prepared.sr_events), raw_space(raw_signal)
    )
    torch.testing.assert_close(
        model.nuisance_function_space(prepared.nuisance_events[: raw_signal.shape[0]]),
        raw_space(raw_signal),
    )
