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


_ONE_D_CASES = [
    pytest.param(
        "issue018_s04_1D_adaptive_neural.json",
        "adaptive_neural",
        "adaptive_neural",
        False,
        id="explicit-adaptive-neural-nuisance",
    ),
    pytest.param(
        "issue018_s04_1D_cubic_binned.json",
        "cubic_bspline",
        "bin_indicators",
        True,
        id="cubic-binned-nuisance",
    ),
    pytest.param(
        "issue018_s04_1D_legendre_binned.json",
        "orthogonal_polynomial",
        "bin_indicators",
        True,
        id="legendre-binned-nuisance",
    ),
    pytest.param(
        "issue018_s04_1D_chebyshev_binned.json",
        "orthogonal_polynomial",
        "bin_indicators",
        True,
        id="chebyshev-binned-nuisance",
    ),
    pytest.param(
        "issue018_s04_1D_sigmoid_binned.json",
        "fixed_sigmoid",
        "bin_indicators",
        True,
        id="fixed-sigmoid-binned-nuisance",
    ),
    pytest.param(
        "issue018_s04_1D_gaussian_binned.json",
        "gaussian_radial_basis",
        "bin_indicators",
        True,
        id="gaussian-radial-basis-binned-nuisance",
    ),
    pytest.param(
        "issue018_s04_1D_cubic_same.json",
        "cubic_bspline",
        "cubic_bspline",
        True,
        id="same-cubic-role-families",
    ),
    pytest.param(
        "issue018_s04_1D_cubic_sigmoid.json",
        "cubic_bspline",
        "fixed_sigmoid",
        True,
        id="different-deterministic-role-families",
    ),
    pytest.param(
        "issue018_s04_1D_adaptive_disabled.json",
        "adaptive_neural",
        None,
        False,
        id="explicit-adaptive-disabled-nuisance",
    ),
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
    fixed_buffers_before = _buffer_snapshot(model)
    coefficients = getattr(model.signal_region_shift_network, "coefficients", None)
    coefficients_before = (
        coefficients.detach().clone() if coefficients is not None else None
    )

    history = model.fit(data_batch)
    _finite_history(history)
    assert model._norm_factor is not None

    fixed_buffers_after = _buffer_snapshot(model)
    assert fixed_buffers_after.keys() == fixed_buffers_before.keys()
    for buffer_name, before in fixed_buffers_before.items():
        torch.testing.assert_close(before, fixed_buffers_after[buffer_name])

    if coefficients_before is not None:
        coefficients_after = model.signal_region_shift_network.coefficients.detach()
        assert not torch.equal(coefficients_before, coefficients_after)

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
    _filename, _f_family, _nuisance_family, _has_coefficients = _case.values
    _ONE_D_PARAMS.append(
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_1D,
                ConfigType.DETECTOR: _DETECTOR_1D,
                ConfigType.TRAIN: Path("test/configs/train") / _filename,
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
def test_issue018_1d_function_space_training_matrix(
    function_execution_context,
    f_family,
    nuisance_family,
    has_coefficients,
    isolated_data_generation,
    detector_effect,
):
    config = function_execution_context.config
    assert config.train__function_space_config.f.family.value == f_family
    if nuisance_family is None:
        assert config.train__function_space_config.nuisance.state.value == "disabled"
    else:
        assert (
            config.train__function_space_config.nuisance.family.value
            == nuisance_family
        )

    data_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = _exercise_model(
        function_execution_context,
        detector_effect,
        data_batch,
        f"issue018_s04_1d_{f_family}",
    )
    assert (
        getattr(model.signal_region_shift_network, "coefficients", None) is not None
    ) is has_coefficients


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_2D,
                ConfigType.DETECTOR: _DETECTOR_2D,
                ConfigType.TRAIN: Path(
                    "test/configs/train/issue018_s04_2D_adaptive_binned.json"
                ),
            },
            id="2d-adaptive-binned",
        ),
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_2D,
                ConfigType.DETECTOR: _DETECTOR_2D,
                ConfigType.TRAIN: Path(
                    "test/configs/train/issue018_s04_2D_cubic_neural.json"
                ),
            },
            id="2d-cubic-neural",
        ),
    ],
    indirect=True,
)
def test_issue018_2d_function_space_training_smoke(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
):
    data_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    _exercise_model(
        function_execution_context,
        detector_effect,
        data_batch,
        "issue018_s04_2d",
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET: _DATASET_1D,
            ConfigType.DETECTOR: _DETECTOR_1D,
            ConfigType.TRAIN: Path(
                "test/configs/train/issue018_s04_1D_adaptive_neural.json"
            ),
        }
    ],
    indirect=True,
)
def test_issue018_matrix_does_not_mutate_loaded_role_config(function_execution_context):
    """Role mappings remain file-loaded values after canonical resolution."""
    config = function_execution_context.config
    before_f = deepcopy(config.train__f)
    before_nuisance = deepcopy(config.train__nuisance)
    resolved = config.train__function_space_config

    assert config.train__f == before_f
    assert config.train__nuisance == before_nuisance
    assert resolved.f.options is not resolved.nuisance.options
