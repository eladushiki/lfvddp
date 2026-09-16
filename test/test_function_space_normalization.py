"""Physical-coordinate geometry must share the model input normalization."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from neural_networks.differentiating_model import DifferentiatingModel
from neural_networks.function_spaces import create_function_space
from neural_networks.nuisance_calculation import BinnedNuisanceCalculation
from train.train_config import TrainConfig


_PHYSICAL_OPTIONS = {
    "cubic_bspline": {"knots": [0.0, 2.5, 5.0, 7.5, 10.0]},
    "orthogonal_polynomial": {
        "basis": "legendre",
        "maximum_degree": 2,
        "domain": [0.0, 10.0],
    },
    "fixed_sigmoid": {"centers": [2.5, 7.5], "widths": [1.75, 1.75]},
    "gaussian_radial_basis": {
        "centers": [2.5, 7.5],
        "widths": [1.75, 1.75],
    },
}


def _physical_batch():
    categories = DataSet.DataSetCategory
    datasets = (
        DataSet(np.array([[0.0], [2.5]]), ["x"], categories.A_SR),
        DataSet(np.array([[7.5], [10.0]]), ["x"], categories.B_SR),
        DataSet(np.array([[1.0], [3.0]]), ["x"], categories.A_CR),
        DataSet(np.array([[7.0], [9.0]]), ["x"], categories.B_CR),
    )
    return DataBatch((dataset, None) for dataset in datasets)


def _model(family, options, nuisance):
    config = TrainConfig(
        train__epochs=1,
        train__number_of_epochs_for_checkpoint=1,
        train__nn_input_dimension=1,
        train__nn_inner_layer_nodes=2,
        train__f={"family": family, "options": options},
        train__nuisance=nuisance,
    )
    return DifferentiatingModel(
        context=SimpleNamespace(config=config),
        detector_effect=SimpleNamespace(),
        is_numerator=True,
        name="physical_geometry",
        dtype=torch.float64,
    )


@pytest.mark.parametrize("family", tuple(_PHYSICAL_OPTIONS))
def test_per_event_geometry_is_configured_in_physical_coordinates(family):
    options = _PHYSICAL_OPTIONS[family]
    model = _model(family, options, {"family": family, "options": options})
    prepared = model._prepare_training_data(_physical_batch())
    raw_sr_events = np.array([[0.0], [2.5], [7.5], [10.0]])
    raw_space = create_function_space("f", family, options, dtype=torch.float64)

    torch.testing.assert_close(
        model.signal_region_shift_network.features(prepared.sr_events),
        raw_space.features(torch.tensor(raw_sr_events, dtype=torch.float64)),
    )
    torch.testing.assert_close(
        model.nuisance_calculation.network.features(prepared.nuisance_data.sr_inputs),
        raw_space.features(torch.tensor(raw_sr_events, dtype=torch.float64)),
    )
    assert model.signal_region_shift_network.geometry == raw_space.geometry


def test_binned_signal_geometry_is_normalized_but_binned_nuisance_stays_physical():
    options = {"minima": [0.0], "maxima": [10.0], "number_of_bins": [4]}
    model = _model("bin_indicators", options, {"family": "bin_indicators", "options": options})
    prepared = model._prepare_training_data(_physical_batch())
    raw_space = create_function_space("f", "bin_indicators", options, dtype=torch.float64)
    for space in (model.signal_region_shift_network, raw_space):
        for index, parameter in enumerate(space._factor_deltas.values()):
            parameter.data.copy_(torch.arange(1, parameter.numel() + 1, dtype=torch.float64))

    raw_sr_events = torch.tensor([[0.0], [2.5], [7.5], [10.0]], dtype=torch.float64)
    torch.testing.assert_close(model.signal_region_shift_network(prepared.sr_events), raw_space(raw_sr_events))
    assert isinstance(model.nuisance_calculation, BinnedNuisanceCalculation)
    np.testing.assert_allclose(
        model.nuisance_calculation.function_space.prediction_grid_edges()[0],
        [0.0, 2.5, 5.0, 7.5, 10.0],
    )
