"""Physical-coordinate geometry must share the model input normalization."""

from pathlib import Path

import numpy as np
import pytest
import torch

from data_tools.data_utils import DataSet
from neural_networks.function_spaces import create_function_space
from neural_networks.nuisance_calculation import (
    BinnedNuisanceCalculation,
    PerEventNuisanceEstimator,
)
from test.environment import ConfigType


_DATASET_CONFIG = Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json")
_DETECTOR_CONFIG = Path("test/configs/detector/basic_1D_detector_config.json")


def _context_params(train_config: str):
    return {
        ConfigType.DATASET: _DATASET_CONFIG,
        ConfigType.DETECTOR: _DETECTOR_CONFIG,
        ConfigType.TRAIN: Path("test/configs/train") / train_config,
    }


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            _context_params("issue018_cubic_bspline_binned.json"),
            id="cubic-bspline",
        ),
        pytest.param(
            _context_params("issue018_orthogonal_legendre_binned.json"),
            id="legendre",
        ),
        pytest.param(
            _context_params("issue018_orthogonal_chebyshev_binned.json"),
            id="chebyshev",
        ),
        pytest.param(
            _context_params("issue018_fixed_sigmoid_binned.json"),
            id="fixed-sigmoid",
        ),
        pytest.param(
            _context_params("issue018_gaussian_radial_basis_binned.json"),
            id="gaussian-radial-basis",
        ),
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
    spec = function_execution_context.config.train__resolved_function_space_config.f
    raw_space = create_function_space("f", spec, dtype=torch.float64)
    raw_sr = np.concatenate(
        (
            data.datasets[DataSet.DataSetCategory.A_SR].events,
            data.datasets[DataSet.DataSetCategory.B_SR].events,
        )
    )

    torch.testing.assert_close(
        model.signal_region_shift_network.features(prepared.sr_events),
        raw_space.features(torch.tensor(raw_sr, dtype=torch.float64)),
    )
    assert model.signal_region_shift_network.geometry == raw_space.geometry


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            _context_params("issue018_cubic_bspline_cubic_bspline.json"),
            id="cubic-bspline-nuisance",
        ),
        pytest.param(
            _context_params("issue018_cubic_bspline_fixed_sigmoid.json"),
            id="fixed-sigmoid-nuisance",
        ),
    ],
    indirect=True,
)
def test_per_event_nuisance_geometry_from_config_uses_the_same_map(
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
    assert isinstance(model.nuisance_calculation, PerEventNuisanceEstimator)
    spec = function_execution_context.config.train__resolved_function_space_config.nuisance
    raw_space = create_function_space("nuisance", spec, dtype=torch.float64)
    raw_sr = np.concatenate(
        (
            data.datasets[DataSet.DataSetCategory.A_SR].events,
            data.datasets[DataSet.DataSetCategory.B_SR].events,
        )
    )

    torch.testing.assert_close(
        model.nuisance_calculation.network.features(prepared.nuisance_data.sr_inputs),
        raw_space.features(torch.tensor(raw_sr, dtype=torch.float64)),
    )
    assert model.nuisance_calculation.network.geometry == raw_space.geometry


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            _context_params("issue018_bin_indicators_binned.json"),
            id="binned-signal-and-nuisance",
        ),
    ],
    indirect=True,
)
def test_binned_signal_geometry_is_normalized_but_binned_nuisance_stays_physical(
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
    spec = function_execution_context.config.train__resolved_function_space_config.f
    raw_space = create_function_space("f", spec, dtype=torch.float64)
    for space in (model.signal_region_shift_network, raw_space):
        for parameter in space._factor_deltas:
            parameter.data.copy_(
                torch.arange(1, parameter.numel() + 1, dtype=torch.float64)
            )

    raw_sr = torch.tensor(
        np.concatenate(
            (
                data.datasets[DataSet.DataSetCategory.A_SR].events,
                data.datasets[DataSet.DataSetCategory.B_SR].events,
            )
        ),
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        model.signal_region_shift_network(prepared.sr_events), raw_space(raw_sr)
    )
    assert isinstance(model.nuisance_calculation, BinnedNuisanceCalculation)
    np.testing.assert_allclose(
        model.nuisance_calculation.function_space.prediction_grid_edges()[0],
        [0.0, 2.5, 5.0, 7.5, 10.0],
    )
