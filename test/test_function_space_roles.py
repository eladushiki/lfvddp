"""The two likelihood roles use independent instances of one space contract."""

from pathlib import Path

import numpy as np
import pytest
import torch

from data_tools.data_utils import DataSet
from neural_networks.differentiating_model import DifferentiatingModel
from neural_networks.function_spaces import BinIndicatorFunction
from test.environment import ConfigType
from train.function_space_config import TrainingBackend, resolve_dual_role_config


_DATASET = Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json")
_DETECTOR = Path("test/configs/detector/basic_1D_detector_config.json")
_TRAIN = Path("test/configs/train")


def _context_params(config_name: str):
    return {
        ConfigType.DATASET: _DATASET,
        ConfigType.DETECTOR: _DETECTOR,
        ConfigType.TRAIN: _TRAIN / config_name,
    }


def test_backend_selection_has_one_source_of_truth():
    resolved = resolve_dual_role_config(
        backend="nplm",
        f={"family": "adaptive_neural", "options": {"input_dimension": 1, "hidden_layer_nodes": 2}},
        nuisance=None,
    )
    assert resolved.backend is TrainingBackend.NPLM


@pytest.mark.parametrize(
    "function_execution_context, expected_nuisance_type",
    [
        pytest.param(_context_params("adaptive_neural_nuisance.json"), None, id="adaptive"),
        pytest.param(_context_params("cubic_bspline_binned.json"), BinIndicatorFunction, id="binned"),
        pytest.param(_context_params("cubic_bspline_fixed_sigmoid.json"), None, id="fixed"),
        pytest.param(_context_params("adaptive_neural_disabled.json"), type(None), id="disabled"),
    ],
    indirect=["function_execution_context"],
)
def test_roles_construct_independent_function_spaces_and_predict(
    function_execution_context,
    expected_nuisance_type,
    isolated_data_generation,
    detector_effect,
):
    batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="independent_roles",
        dtype=torch.float64,
    )
    prepared = model._prepare_training_data(batch)
    assert model.signal_region_shift_network is not None
    if expected_nuisance_type is type(None):
        assert model.nuisance_function_space is None
    else:
        assert model.nuisance_function_space is not None
        if expected_nuisance_type is not None:
            assert isinstance(model.nuisance_function_space, expected_nuisance_type)
        assert {
            id(parameter) for parameter in model.signal_region_shift_network.parameters()
        }.isdisjoint({id(parameter) for parameter in model.nuisance_function_space.parameters()})

    loss = model(prepared)
    assert torch.isfinite(loss)
    data = batch.datasets[DataSet.DataSetCategory.A_SR]
    primary = model.predict(data)
    secondary = model.predict_secondary(data)
    nuisance = model.predict_theta(data)
    assert primary.shape == secondary.shape == nuisance.shape == (data.n_samples, 1)
    assert np.isfinite(primary).all() and np.isfinite(secondary).all()


@pytest.mark.parametrize(
    "function_execution_context",
    [pytest.param(_context_params("bin_indicators_binned.json"), id="binned")],
    indirect=True,
)
def test_binned_nuisance_has_independent_cartesian_cell_gradients(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
):
    batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="binned_nuisance_gradients",
        dtype=torch.float64,
    )
    prepared = model._prepare_training_data(batch)
    assert isinstance(model.nuisance_function_space, BinIndicatorFunction)
    loss = model(prepared)
    loss.backward()
    assert model.nuisance_function_space.coefficients.grad is not None
