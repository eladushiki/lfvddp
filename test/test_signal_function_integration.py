import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from data_tools.data_utils import DataSet
from neural_networks.differentiating_model import DifferentiatingModel
from test.environment import ConfigType


_DATASET_1D = Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json")
_DETECTOR_1D = Path("test/configs/detector/basic_1D_detector_config.json")
_DATASET_2D = Path("test/configs/dataset/disjoint_2D_generated_dataset_config.json")
_DETECTOR_2D = Path("test/configs/detector/basic_2D_detector_config.json")
_TRAIN_CONFIG_DIR = Path("test/configs/train")


_ONE_D_CASES = [
    pytest.param(_TRAIN_CONFIG_DIR / "adaptive_neural_nuisance.json", "adaptive_neural", "adaptive_neural", False, id="explicit-adaptive-neural-nuisance"),
    pytest.param(_TRAIN_CONFIG_DIR / "cubic_bspline_binned.json", "cubic_bspline", "bin_indicators", True, id="cubic-binned-nuisance"),
    pytest.param(_TRAIN_CONFIG_DIR / "orthogonal_legendre_binned.json", "orthogonal_polynomial", "bin_indicators", True, id="legendre-binned-nuisance"),
    pytest.param(_TRAIN_CONFIG_DIR / "orthogonal_chebyshev_binned.json", "orthogonal_polynomial", "bin_indicators", True, id="chebyshev-binned-nuisance"),
    pytest.param(_TRAIN_CONFIG_DIR / "fixed_sigmoid_binned.json", "fixed_sigmoid", "bin_indicators", True, id="fixed-sigmoid-binned-nuisance"),
    pytest.param(_TRAIN_CONFIG_DIR / "gaussian_radial_basis_binned.json", "gaussian_radial_basis", "bin_indicators", True, id="gaussian-radial-basis-binned-nuisance"),
    pytest.param(_TRAIN_CONFIG_DIR / "cubic_bspline_cubic_bspline.json", "cubic_bspline", "cubic_bspline", True, id="same-cubic-role-families"),
    pytest.param(_TRAIN_CONFIG_DIR / "cubic_bspline_fixed_sigmoid.json", "cubic_bspline", "fixed_sigmoid", True, id="different-deterministic-role-families"),
    pytest.param(_TRAIN_CONFIG_DIR / "adaptive_neural_disabled.json", "adaptive_neural", None, False, id="explicit-adaptive-disabled-nuisance"),
]


def _finite_history(history):
    assert history["loss"]
    assert len(history["loss"]) == len(history["epoch"])
    assert all(
        math.isfinite(float(value)) for value in history["loss"]
    ), history["loss"]


def _buffer_snapshot(model):
    return {
        name: value.detach().clone()
        for name, value in model.named_buffers()
    }


def _exercise_model(context, detector_effect, data_batch, name):
    model = DifferentiatingModel(
        context=context,
        detector_effect=detector_effect,
        is_numerator=True,
        name=f"{name}_numerator",
        dtype=torch.float64,
        device="cpu",
    )
    # Geometry is transformed once from its physical configuration coordinates
    # using the same batch affine map as the model inputs.
    model._prepare_training_data(data_batch)
    fixed_buffers_before = _buffer_snapshot(model)

    history = model.fit(data_batch)
    _finite_history(history)
    assert model._norm_factor is not None

    fixed_buffers_after = _buffer_snapshot(model)
    assert fixed_buffers_after.keys() == fixed_buffers_before.keys()
    for buffer_name, before in fixed_buffers_before.items():
        torch.testing.assert_close(before, fixed_buffers_after[buffer_name])

    prediction_data = data_batch.datasets[DataSet.DataSetCategory.A_SR]
    predictions = (
        model.predict(prediction_data),
        model.predict_secondary(prediction_data),
        model.predict_theta(prediction_data),
    )
    expected_shape = (prediction_data.n_samples, 1)
    assert tuple(prediction.shape for prediction in predictions) == (
        expected_shape,
        expected_shape,
        expected_shape,
    )
    assert all(
        math.isfinite(float(value))
        for prediction in predictions
        for value in prediction.flat
    )

    denominator = DifferentiatingModel(
        context=context,
        detector_effect=detector_effect,
        is_numerator=False,
        name=f"{name}_denominator",
        dtype=torch.float64,
        device="cpu",
    )
    denominator_history = denominator.calculate_loss_statically(data_batch)
    _finite_history(denominator_history)
    denominator_predictions = (
        denominator.predict(prediction_data),
        denominator.predict_secondary(prediction_data),
        denominator.predict_theta(prediction_data),
    )
    assert tuple(prediction.shape for prediction in denominator_predictions) == (
        expected_shape,
        expected_shape,
        expected_shape,
    )
    assert all(
        math.isfinite(float(value))
        for prediction in denominator_predictions
        for value in prediction.flat
    )

    return model


_ONE_D_PARAMS = []
for _case in _ONE_D_CASES:
    _train_fixture, _f_family, _nuisance_family, _has_coefficients = _case.values
    _ONE_D_PARAMS.append(
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_1D,
                ConfigType.DETECTOR: _DETECTOR_1D,
                ConfigType.TRAIN: _train_fixture,
            },
            _f_family,
            _nuisance_family,
            _has_coefficients,
            id=_case.id,
        )
    )


@pytest.mark.parametrize(
    ("function_execution_context", "f_family", "nuisance_family", "has_coefficients"),
    _ONE_D_PARAMS,
    indirect=["function_execution_context"],
)
def test_function_space_1d_training_matrix(
    function_execution_context,
    f_family,
    nuisance_family,
    has_coefficients,
    isolated_data_generation,
    detector_effect,
):
    config = function_execution_context.config
    resolved = config.resolve_function_space_config()
    assert resolved.f.family == f_family
    if nuisance_family is None:
        assert resolved.nuisance is None
    else:
        assert resolved.nuisance is not None
        assert resolved.nuisance.family == nuisance_family

    data_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = _exercise_model(
        function_execution_context,
        detector_effect,
        data_batch,
        f"function_space_1d_{f_family}",
    )
    has_coefficients_after_training = any(
        name == "coefficients" for name, _ in model.signal_region_shift_network.named_parameters()
    )
    assert has_coefficients_after_training is has_coefficients


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_2D,
                ConfigType.DETECTOR: _DETECTOR_2D,
                ConfigType.TRAIN: _TRAIN_CONFIG_DIR / "two_dimensional_adaptive_neural_binned.json",
            },
            id="2d-adaptive-binned",
        ),
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_2D,
                ConfigType.DETECTOR: _DETECTOR_2D,
                ConfigType.TRAIN: _TRAIN_CONFIG_DIR / "two_dimensional_cubic_bspline_adaptive_neural.json",
            },
            id="2d-cubic-neural",
        ),
    ],
    indirect=True,
)
def test_function_space_2d_training_smoke(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
):
    data_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    _exercise_model(
        function_execution_context,
        detector_effect,
        data_batch,
        "function_space_2d",
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET: _DATASET_1D,
            ConfigType.DETECTOR: _DETECTOR_1D,
            ConfigType.TRAIN: _TRAIN_CONFIG_DIR / "adaptive_neural_nuisance.json",
        }
    ],
    indirect=True,
)
def test_function_space_matrix_does_not_mutate_loaded_role_config(function_execution_context):
    """Role mappings remain file-loaded values after canonical resolution."""
    config = function_execution_context.config
    before_f = deepcopy(config.train__f)
    before_nuisance = deepcopy(config.train__nuisance)
    resolved = config.resolve_function_space_config()

    assert config.train__f == before_f
    assert config.train__nuisance == before_nuisance
    assert resolved.f.options is not resolved.nuisance.options
