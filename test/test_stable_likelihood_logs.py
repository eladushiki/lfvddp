"""Raw-output log factors retain derivatives lost by rounded tanh values."""

from math import exp
from pathlib import Path

import pytest
import torch

from neural_networks.function_spaces import create_function_space
from neural_networks.likelihood_parameterization import (
    LIKELIHOOD_SHIFT_BOUND,
    bounded_likelihood_shift_and_logs,
    smoothly_bounded_likelihood_shift,
)
from test.environment import ConfigType
from test.function_space_cases import FUNCTION_SPACE_OPTIONS, NEURAL_OUTPUT_CONFIGS
from train.function_space_config import FunctionSpaceSpec


@pytest.mark.parametrize("mapping", ["tanh", "algebraic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_likelihood_log_values_and_gradients_match_ordinary_region(mapping, dtype):
    raw = torch.tensor([-2.0, -0.2, 0.0, 0.2, 2.0], dtype=dtype, requires_grad=True)
    shift, positive, negative = bounded_likelihood_shift_and_logs(raw, mapping)
    expected_shift = smoothly_bounded_likelihood_shift(raw, mapping)
    torch.testing.assert_close(shift, expected_shift, rtol=0, atol=0)
    for actual, expected in (
        (positive, torch.log1p(expected_shift)),
        (negative, torch.log1p(-expected_shift)),
    ):
        tolerance = 1e-6 if dtype == torch.float32 else 1e-14
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        actual_grad = torch.autograd.grad(actual.sum(), raw, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected.sum(), raw, retain_graph=True)[0]
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=tolerance, rtol=tolerance
        )


@pytest.mark.parametrize("raw_value", [-103.0, -60.0, -20.0, 20.0, 60.0, 103.0])
def test_tanh_log_factors_retain_analytic_gradient_after_tanh_rounds(raw_value):
    raw = torch.tensor(raw_value, dtype=torch.float64, requires_grad=True)
    shift, positive, negative = bounded_likelihood_shift_and_logs(raw, "tanh")
    assert abs(torch.tanh(raw)) == 1
    assert abs(shift) == LIKELIHOOD_SHIFT_BOUND
    tail = exp(-2 * abs(raw_value))
    slope = 4 * LIKELIHOOD_SHIFT_BOUND * tail / (1 + tail) ** 2
    for sign, logarithm in [(1, positive), (-1, negative)]:
        factor = (
            1 - LIKELIHOOD_SHIFT_BOUND + 2 * LIKELIHOOD_SHIFT_BOUND * tail / (1 + tail)
            if sign * raw_value < 0
            else 1
            + LIKELIHOOD_SHIFT_BOUND
            - 2 * LIKELIHOOD_SHIFT_BOUND * tail / (1 + tail)
        )
        expected = raw.new_tensor(sign * slope / factor)
        actual = torch.autograd.grad(logarithm, raw, retain_graph=True)[0]
        assert actual != 0
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0)


def test_tanh_log_gradient_survives_float32_rounding():
    raw = torch.tensor(-20.0, dtype=torch.float32, requires_grad=True)
    _, positive, _ = bounded_likelihood_shift_and_logs(raw, "tanh")
    assert torch.autograd.grad(positive, raw)[0] > 0
    rounded_log = torch.log1p(smoothly_bounded_likelihood_shift(raw, "tanh"))
    assert torch.autograd.grad(rounded_log, raw)[0] == 0


@pytest.mark.parametrize("mapping", ["tanh", "algebraic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_factors_remain_finite_at_extreme_finite_raw_values(mapping, dtype):
    maximum = torch.finfo(dtype).max
    raw = torch.tensor([-maximum, 0.0, maximum], dtype=dtype)
    outputs = bounded_likelihood_shift_and_logs(raw, mapping)
    assert all(torch.isfinite(value).all() for value in outputs)
    torch.testing.assert_close(outputs[1], outputs[2].flip(0), rtol=0, atol=0)


@pytest.mark.parametrize("family, options", FUNCTION_SPACE_OPTIONS.items())
def test_function_space_likelihood_path_preserves_predictions(family, options):
    space = create_function_space(
        FunctionSpaceSpec(family, options), dtype=torch.float64, observable_names=("x",)
    )
    space.initialize_parameters(gain=0.2)
    events = torch.tensor([[0.25], [0.75], [1.25]], dtype=torch.float64)
    shift, positive, negative = space.forward_with_log_factors(events)
    torch.testing.assert_close(shift, space(events), rtol=0, atol=0)
    torch.testing.assert_close(positive, torch.log1p(shift), rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(negative, torch.log1p(-shift), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize(
    "function_execution_context", [NEURAL_OUTPUT_CONFIGS[1]], indirect=True
)
@pytest.mark.parametrize("is_numerator", [True, False])
@pytest.mark.parametrize("raw_value", [-20.0, 20.0])
def test_training_uses_stable_logs_for_both_neural_roles(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
    is_numerator,
    raw_value,
):
    model = differentiating_model_factory(
        function_execution_context, detector_effect, is_numerator=is_numerator
    )
    batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    data = model._prepare_training_data(batch)
    spaces = [model.nuisance_function_space]
    if is_numerator:
        spaces.append(model.signal_region_shift_network)
    with torch.no_grad():
        for space in spaces:
            space.output.weight.zero_()
            space.output.bias.fill_(raw_value)
    loss = model(data)
    assert torch.isfinite(loss)
    loss.backward()
    for space in spaces:
        assert torch.isfinite(space.output.bias.grad).all()
        assert torch.count_nonzero(space.output.bias.grad)


@pytest.mark.parametrize(
    "function_execution_context",
    [
        *NEURAL_OUTPUT_CONFIGS,
        *[
            {
                **NEURAL_OUTPUT_CONFIGS[0],
                ConfigType.TRAIN: Path(f"test/configs/train/{name}.json"),
            }
            for name in ("adaptive_neural_disabled", "cubic_bspline_binned")
        ],
    ],
    indirect=True,
)
@pytest.mark.parametrize("is_numerator", [True, False])
def test_training_log_factors_match_original_likelihood_and_gradients(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
    is_numerator,
):
    model = differentiating_model_factory(
        function_execution_context, detector_effect, is_numerator=is_numerator
    )
    batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    data = model._prepare_training_data(batch)
    actual = model(data)
    signal = (
        model.signal_region_shift_network(data.sr_events).reshape(-1)
        if is_numerator
        else data.sr_events.new_zeros(data.N_sr)
    )
    nuisance = (
        model.nuisance_function_space(data.nuisance_events).reshape(-1)
        if model.nuisance_function_space is not None
        else data.nuisance_events.new_zeros(data.nuisance_events.shape[0])
    )
    expected = model._assemble_loss(
        signal_hypothesis_sr_shift=signal,
        nuisance_estimates=tuple(
            nuisance.split((data.N_a_sr, data.N_b_sr, data.N_a_cr, data.N_b_cr))
        ),
        data=data,
    )
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-12)
    parameters = tuple(model.parameters())
    if parameters:
        actual_gradients = torch.autograd.grad(actual, parameters)
        expected_gradients = torch.autograd.grad(expected, parameters)
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, atol=1e-10, rtol=1e-12
            )
