from pathlib import Path

import numpy as np
import pytest
import torch

from data_tools.data_utils import DataSet
from frame.command_line.handle_args import create_config_from_paths
from frame.file_system.training_history import HistoryKeys
from neural_networks.differentiating_model import (
    DifferentiatingModel,
    _SignalRegionShiftEstimator,
    _PreparedTrainingData,
)
from neural_networks.nuisance_calculation import (
    NeuralPerEventNuisanceEstimator,
    NuisanceEvaluation,
    ScalarBinnedNuisanceEstimator,
    WeightedNuisanceValues,
    _ThetaEstimator,
)
from test.environment import DEFAULT_CONFIG_PATHS, ConfigType
from train.checkpoints import (
    _torch_load,
    save_training_checkpoint,
)
from train.model_trainer import SequentialTrainLauncher
from train.runtime_resources import RuntimeAllocation
from train.tensorboard_clutch import log_t_history


def test_theta_estimator_matches_network_dimensions_and_bounds_output():
    estimator = _ThetaEstimator(
        input_dimension=2,
        hidden_size=2,
        output_dimension=1,
        dtype=torch.float64,
    )

    theta = estimator(torch.zeros((3, 2), dtype=torch.float64))

    assert theta.shape == (3,)
    assert torch.all(torch.abs(theta) < 1)


ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG = {
    ConfigType.DATASET: Path(
        "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
    ),
    ConfigType.DETECTOR: Path(
        "test/configs/detector/basic_1D_detector_config.json"
    ),
    ConfigType.TRAIN: Path(
        "test/configs/train/short_1D_train_config_without_nuisance.json"
    ),
}
ONE_DIMENSION_WITH_NEURAL_NUISANCE_CONFIG = {
    **ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG,
    ConfigType.TRAIN: Path(
        "test/configs/train/short_1D_train_config_with_neural_nuisance.json"
    ),
}


@pytest.mark.parametrize(
    "train_config_path, error_message",
    [
        (
            "test/configs/train/short_1D_train_config_with_mixed_binned_nuisance.json",
            "Binned nuisance configuration must not define",
        ),
        (
            "test/configs/train/short_1D_train_config_with_mixed_neural_nuisance.json",
            "Neural nuisance configuration must not define",
        ),
    ],
)
def test_mixed_nuisance_configurations_are_rejected(
    train_config_path, error_message
):
    config_paths = {
        **DEFAULT_CONFIG_PATHS,
        ConfigType.TRAIN: Path(train_config_path),
    }

    with pytest.raises(ValueError, match=error_message):
        create_config_from_paths(list(config_paths.values()))
ONE_DIMENSION_WITH_ADAPTIVE_LEARNING_RATE_CONFIG = {
    **ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG,
    ConfigType.TRAIN.value: Path(
        "test/configs/train/short_1D_train_config_with_adaptive_learning_rate.json"
    ),
}
ONE_DIMENSION_WITH_INCREASING_LEARNING_RATE_CONFIG = {
    **ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG,
    ConfigType.TRAIN.value: Path(
        "test/configs/train/short_1D_train_config_with_increasing_learning_rate.json"
    ),
}
TWO_DIMENSION_WITH_NUISANCE_CONFIG = {
    ConfigType.DATASET.value: Path(
        "test/configs/dataset/disjoint_2D_generated_dataset_config.json"
    ),
    ConfigType.DETECTOR.value: Path(
        "test/configs/detector/basic_2D_detector_config.json"
    ),
    ConfigType.TRAIN.value: Path(
        "test/configs/train/short_2D_train_config_with_nuisance.json"
    ),
}


def _train_numerator(function_execution_context, data_batch, detector_effect, name):
    # Unit training stays on one CPU without depending on the developer
    # machine's resources.
    allocation = RuntimeAllocation(
        cpu_count=1,
        cpu_affinity=(),
        assigned_gpu_ids=(),
        visible_gpu_count=0,
        gpu_names=(),
        gpu_total_memory_bytes=(),
    )
    train_launcher = SequentialTrainLauncher(
        function_execution_context,
        detector_effect,
        allocation=allocation,
    )
    train_idx = train_launcher.add_training(
        data_batch=data_batch,
        detector_effect=detector_effect,
        is_numerator=True,
        name=name,
    )
    train_launcher.execute_trainings()
    return train_launcher.get_train_result(train_idx)


def _reference_loss(
    f_of_x_sr,
    theta_of_x_sr,
    theta_of_x_cr,
    number_of_a_sr_events,
    number_of_b_sr_events,
    number_of_a_cr_events,
    number_of_b_cr_events,
):
    zero_a_cr = f_of_x_sr.new_zeros(number_of_a_cr_events)
    zero_b_cr = f_of_x_sr.new_zeros(number_of_b_cr_events)
    f = torch.cat(
        (
            f_of_x_sr[:number_of_a_sr_events],
            zero_a_cr,
            f_of_x_sr[number_of_a_sr_events:],
            zero_b_cr,
        )
    )
    theta = torch.cat(
        (
            theta_of_x_sr[:number_of_a_sr_events],
            theta_of_x_cr[:number_of_a_cr_events],
            theta_of_x_sr[number_of_a_sr_events:],
            theta_of_x_cr[number_of_a_cr_events:],
        )
    )
    category_sizes = (
        number_of_a_sr_events,
        number_of_a_cr_events,
        number_of_b_sr_events,
        number_of_b_cr_events,
    )
    category_masks = []
    offset = 0
    for category_size in category_sizes:
        mask = torch.zeros_like(theta, dtype=torch.bool)
        mask[offset : offset + category_size] = True
        category_masks.append(mask)
        offset += category_size
    a_sr_mask, a_cr_mask, b_sr_mask, b_cr_mask = category_masks
    sr_mask = a_sr_mask | b_sr_mask
    cr_mask = a_cr_mask | b_cr_mask
    theta_a = theta * (a_sr_mask | a_cr_mask)
    theta_b = theta * (b_sr_mask | b_cr_mask)
    theta_sr = theta * sr_mask
    theta_cr = theta * cr_mask
    sr_term = (
        number_of_a_sr_events * (1 + f) * (1 + theta_sr) * sr_mask
        + number_of_b_sr_events * (1 - f) * (1 - theta_sr) * sr_mask
    ) / (number_of_a_sr_events + number_of_b_sr_events)
    cr_term = (
        number_of_a_cr_events * (1 + theta_cr) * cr_mask
        + number_of_b_cr_events * (1 - theta_cr) * cr_mask
    ) / (number_of_a_cr_events + number_of_b_cr_events)
    return (
        sr_term
        + cr_term
        - torch.log1p(f) * a_sr_mask
        - torch.log1p(-f) * b_sr_mask
        - torch.log(1 + theta_a)
        - torch.log(1 - theta_b)
    ).sum()


def _assemble_compact_loss_for_test(
    signal_region_shift,
    theta_sr,
    theta_cr,
    a_cr_multiplicities,
    b_cr_multiplicities,
    number_of_a_sr,
    number_of_b_sr,
    number_of_a_cr,
    number_of_b_cr,
):
    number_of_sr = number_of_a_sr + number_of_b_sr
    number_of_cr = number_of_a_cr + number_of_b_cr
    control_region_linear_nuisance_coefficient = (
        number_of_a_cr - number_of_b_cr
    ) / number_of_cr
    data = _PreparedTrainingData(
        sr_events=torch.empty((number_of_sr, 0), dtype=theta_sr.dtype),
        nuisance_data=None,
        N_a_sr=number_of_a_sr,
        N_b_sr=number_of_b_sr,
        N_a_cr=number_of_a_cr,
        N_b_cr=number_of_b_cr,
        sr_category_imbalance=(
            number_of_a_sr / number_of_sr - number_of_b_sr / number_of_sr
        ),
        nuisance_cr_coefficient=(
            control_region_linear_nuisance_coefficient
        ),
    )
    nuisance = NuisanceEvaluation(
        nuisance_sr_values=theta_sr,
        nuisance_cr_a=WeightedNuisanceValues(theta_cr, a_cr_multiplicities),
        nuisance_cr_b=WeightedNuisanceValues(theta_cr, b_cr_multiplicities),
    )
    return DifferentiatingModel._assemble_loss(
        signal_hypothesis_sr_shift=signal_region_shift,
        nuisance_estimates=nuisance,
        data=data,
    )


@pytest.mark.parametrize("with_nuisance", [False, True])
@pytest.mark.parametrize("category_sizes", [(2, 3, 4, 5), (4, 2, 3, 6)])
@pytest.mark.parametrize("input_dimension", [1, 2, 4])
def test_compact_loss_matches_full_event_value_and_gradients(
    with_nuisance, category_sizes, input_dimension
):
    number_of_a_sr, number_of_a_cr, number_of_b_sr, number_of_b_cr = category_sizes
    number_of_sr = number_of_a_sr + number_of_b_sr
    number_of_cr = number_of_a_cr + number_of_b_cr
    events = torch.linspace(
        -0.8, 0.9, number_of_sr * input_dimension, dtype=torch.float64
    ).reshape(number_of_sr, input_dimension)
    f_parameters = torch.linspace(
        -0.4, 0.6, input_dimension, dtype=torch.float64, requires_grad=True
    )
    f = events @ f_parameters
    if with_nuisance:
        theta_sr = torch.linspace(
            -1.0 + 2e-6,
            1.0 - 2e-6,
            number_of_sr,
            dtype=torch.float64,
            requires_grad=True,
        )
        a_cr_bin_counts = torch.tensor([1, number_of_a_cr - 1, 0], dtype=torch.float64)
        b_cr_bin_counts = torch.tensor([0, number_of_b_cr - 1, 1], dtype=torch.float64)
        theta_cr_bins = torch.tensor(
            [-1.0 + 3e-6, 0.35, 1.0 - 3e-6],
            dtype=torch.float64,
            requires_grad=True,
        )
        theta_cr = torch.cat(
            (
                torch.repeat_interleave(theta_cr_bins, a_cr_bin_counts.long()),
                torch.repeat_interleave(theta_cr_bins, b_cr_bin_counts.long()),
            )
        )
        actual = _assemble_compact_loss_for_test(
            f,
            theta_sr,
            theta_cr_bins,
            a_cr_bin_counts,
            b_cr_bin_counts,
            number_of_a_sr,
            number_of_b_sr,
            number_of_a_cr,
            number_of_b_cr,
        )
    else:
        theta_sr, theta_cr = (
            torch.zeros(number_of_sr, dtype=torch.float64),
            torch.zeros(number_of_cr, dtype=torch.float64),
        )
        actual = _assemble_compact_loss_for_test(
            f,
            theta_sr,
            theta_cr,
            torch.cat(
                (
                    torch.ones(number_of_a_cr, dtype=torch.float64),
                    torch.zeros(number_of_b_cr, dtype=torch.float64),
                )
            ),
            torch.cat(
                (
                    torch.zeros(number_of_a_cr, dtype=torch.float64),
                    torch.ones(number_of_b_cr, dtype=torch.float64),
                )
            ),
            number_of_a_sr,
            number_of_b_sr,
            number_of_a_cr,
            number_of_b_cr,
        )
    expected = _reference_loss(
        f,
        theta_sr,
        theta_cr,
        number_of_a_sr,
        number_of_b_sr,
        number_of_a_cr,
        number_of_b_cr,
    )
    actual_gradients = torch.autograd.grad(
        actual,
        (f_parameters,)
        if not with_nuisance
        else (f_parameters, theta_sr, theta_cr_bins),
        retain_graph=True,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (f_parameters,)
        if not with_nuisance
        else (f_parameters, theta_sr, theta_cr_bins),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("category_sizes", [(2, 3, 4, 5), (4, 2, 3, 6)])
def test_compact_nuisance_denominator_matches_full_event_gradients(category_sizes):
    number_of_a_sr, number_of_a_cr, number_of_b_sr, number_of_b_cr = category_sizes
    number_of_sr = number_of_a_sr + number_of_b_sr
    theta_sr = torch.linspace(
        -1.0 + 2e-6,
        1.0 - 2e-6,
        number_of_sr,
        dtype=torch.float64,
        requires_grad=True,
    )
    a_cr_bin_counts = torch.tensor([1, number_of_a_cr - 1, 0], dtype=torch.float64)
    b_cr_bin_counts = torch.tensor([0, number_of_b_cr - 1, 1], dtype=torch.float64)
    theta_cr_bins = torch.tensor(
        [-1.0 + 3e-6, 0.35, 1.0 - 3e-6],
        dtype=torch.float64,
        requires_grad=True,
    )
    theta_cr = torch.cat(
        (
            torch.repeat_interleave(theta_cr_bins, a_cr_bin_counts.long()),
            torch.repeat_interleave(theta_cr_bins, b_cr_bin_counts.long()),
        )
    )
    zeros = torch.zeros(number_of_sr, dtype=torch.float64)
    actual = _assemble_compact_loss_for_test(
        None,
        theta_sr,
        theta_cr_bins,
        a_cr_bin_counts,
        b_cr_bin_counts,
        number_of_a_sr,
        number_of_b_sr,
        number_of_a_cr,
        number_of_b_cr,
    )
    expected = _reference_loss(
        zeros,
        theta_sr,
        theta_cr,
        number_of_a_sr,
        number_of_b_sr,
        number_of_a_cr,
        number_of_b_cr,
    )
    actual_gradients = torch.autograd.grad(
        actual, (theta_sr, theta_cr_bins), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, (theta_sr, theta_cr_bins))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12
        )


def test_no_nuisance_loss_is_bitwise_equivalent_to_explicit_zero_nuisance():
    number_of_a_sr = 25_019
    number_of_b_sr = 25_255
    number_of_sr = number_of_a_sr + number_of_b_sr
    signal_region_shift = torch.linspace(
        -0.8,
        0.8,
        number_of_sr,
        dtype=torch.float64,
        requires_grad=True,
    )
    data = _PreparedTrainingData(
        sr_events=torch.empty((number_of_sr, 1), dtype=torch.float64),
        nuisance_data=None,
        N_a_sr=number_of_a_sr,
        N_b_sr=number_of_b_sr,
        N_a_cr=24_900,
        N_b_cr=25_100,
        sr_category_imbalance=(
            number_of_a_sr / number_of_sr - number_of_b_sr / number_of_sr
        ),
        nuisance_cr_coefficient=-0.004,
    )
    empty = torch.empty(0, dtype=torch.float64)
    explicit_zero_nuisance = NuisanceEvaluation(
        nuisance_sr_values=torch.zeros(number_of_sr, dtype=torch.float64),
        nuisance_cr_a=WeightedNuisanceValues(empty),
        nuisance_cr_b=WeightedNuisanceValues(empty),
    )

    optimized = DifferentiatingModel._assemble_loss(
        signal_hypothesis_sr_shift=signal_region_shift,
        nuisance_estimates=None,
        data=data,
    )
    reference = DifferentiatingModel._assemble_loss(
        signal_hypothesis_sr_shift=signal_region_shift,
        nuisance_estimates=explicit_zero_nuisance,
        data=data,
    )
    optimized_gradient = torch.autograd.grad(
        optimized, signal_region_shift, retain_graph=True
    )[0]
    reference_gradient = torch.autograd.grad(reference, signal_region_shift)[0]

    assert torch.equal(optimized, reference)
    assert torch.equal(optimized_gradient, reference_gradient)


def test_balanced_cr_skips_zero_weighted_dot_products(monkeypatch):
    dot_calls = 0
    original_dot = torch.dot

    def count_dot_calls(*args, **kwargs):
        nonlocal dot_calls
        dot_calls += 1
        return original_dot(*args, **kwargs)

    monkeypatch.setattr(torch, "dot", count_dot_calls)
    _assemble_compact_loss_for_test(
        signal_region_shift=torch.linspace(-0.2, 0.2, 6, dtype=torch.float64),
        theta_sr=torch.linspace(-0.1, 0.1, 6, dtype=torch.float64),
        theta_cr=torch.tensor([-0.2, 0.3], dtype=torch.float64),
        a_cr_multiplicities=torch.tensor([2.0, 3.0], dtype=torch.float64),
        b_cr_multiplicities=torch.tensor([1.0, 4.0], dtype=torch.float64),
        number_of_a_sr=3,
        number_of_b_sr=3,
        number_of_a_cr=5,
        number_of_b_cr=5,
    )

    # One SR interaction and two CR log terms remain. The two CR linear dots
    # have an exactly zero coefficient and must never enter the autograd graph.
    assert dot_calls == 3


def test_cpu_thread_count_preserves_loss_and_gradient_bits():
    number_of_a_sr = 25_019
    number_of_b_sr = 25_255
    number_of_sr = number_of_a_sr + number_of_b_sr
    base_signal = torch.linspace(-0.8, 0.8, number_of_sr, dtype=torch.float64)
    base_theta_sr = torch.linspace(-0.7, 0.7, number_of_sr, dtype=torch.float64)
    base_theta_cr = torch.linspace(-0.6, 0.6, 32, dtype=torch.float64)
    a_cr_multiplicities = torch.arange(1, 33, dtype=torch.float64)
    b_cr_multiplicities = torch.arange(32, 0, -1, dtype=torch.float64)

    def loss_and_gradients(number_of_threads):
        torch.set_num_threads(number_of_threads)
        signal = base_signal.clone().requires_grad_()
        theta_sr = base_theta_sr.clone().requires_grad_()
        theta_cr = base_theta_cr.clone().requires_grad_()
        loss = _assemble_compact_loss_for_test(
            signal,
            theta_sr,
            theta_cr,
            a_cr_multiplicities,
            b_cr_multiplicities,
            number_of_a_sr,
            number_of_b_sr,
            530,
            526,
        )
        gradients = torch.autograd.grad(loss, (signal, theta_sr, theta_cr))
        return loss.detach(), tuple(gradient.detach() for gradient in gradients)

    original_thread_count = torch.get_num_threads()
    try:
        capped = loss_and_gradients(4)
        uncapped = loss_and_gradients(8)
    finally:
        torch.set_num_threads(original_thread_count)

    assert torch.equal(capped[0], uncapped[0])
    assert all(
        torch.equal(capped_gradient, uncapped_gradient)
        for capped_gradient, uncapped_gradient in zip(capped[1], uncapped[1])
    )


@pytest.mark.parametrize("input_dimension", [1, 2, 4])
def test_signal_region_shift_estimator_clamps_result(input_dimension):
    estimator = _SignalRegionShiftEstimator(
        input_dimension=input_dimension,
        hidden_size=4,
        output_dimension=1,
        dtype=torch.float64,
    )
    with torch.no_grad():
        estimator.output.weight.fill_(2)
        estimator.output.bias.fill_(-2)

    estimate = estimator(torch.ones((7, input_dimension), dtype=torch.float64))
    assert estimate.shape == (7, 1)
    assert torch.all(estimate.abs() < 1)


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_NEURAL_NUISANCE_CONFIG],
    indirect=True,
)
def test_neural_theta_preparation_skips_detector_bin_compression(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="neural_theta_model",
    )

    prepared = model._prepare_training_data(detected_batch)

    assert isinstance(model.nuisance_calculation, NeuralPerEventNuisanceEstimator)
    assert prepared.nuisance_data.cr_inputs is not None
    assert prepared.nuisance_data.cr_inputs.shape[0] == prepared.number_of_cr_events
    assert prepared.nuisance_data.number_of_a_cr_events == prepared.N_a_cr
    nuisance_evaluation = model.nuisance_calculation.evaluate(prepared.nuisance_data)
    assert nuisance_evaluation.nuisance_cr_a.values.shape[0] == prepared.N_a_cr
    assert nuisance_evaluation.nuisance_cr_b.values.shape[0] == prepared.N_b_cr
    assert nuisance_evaluation.nuisance_cr_a.weights is None
    assert nuisance_evaluation.nuisance_cr_b.weights is None

    loss = model(prepared)
    assert torch.isfinite(loss)
    loss.backward()

    _, model._norm_factor = detected_batch.get_normalized()
    prediction_data = detected_batch.datasets[DataSet.DataSetCategory.A_SR]
    theta_prediction = model.predict_theta(prediction_data)
    assert theta_prediction.shape == (prediction_data.n_samples, 1)
    assert np.isfinite(theta_prediction).all()


@pytest.mark.parametrize(
    "function_execution_context",
    [TWO_DIMENSION_WITH_NUISANCE_CONFIG],
    indirect=True,
)
def test_nuisance_preparation_compresses_cr_and_uses_one_theta_evaluation(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="compressed_nuisance_model",
    )
    prepared = model._prepare_training_data(detected_batch)

    assert isinstance(model.nuisance_calculation, ScalarBinnedNuisanceEstimator)
    assert prepared.nuisance_data.nuisance_cr_bin_indices is not None
    assert int(prepared.nuisance_data.nuisance_cr_a_multiplicities.sum()) == (
        prepared.N_a_cr
    )
    assert int(prepared.nuisance_data.nuisance_cr_b_multiplicities.sum()) == (
        prepared.N_b_cr
    )
    assert (
        prepared.nuisance_data.nuisance_cr_bin_indices.shape[0]
        <= prepared.number_of_cr_events
    )

    loss = model(prepared)
    loss.backward()

    assert all(
        nuisance_parameter.grad is not None
        for nuisance_parameter in model.nuisance_calculation.parameters()
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG],
    indirect=True,
)
def test_static_denominator_behavior(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    denominator = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=False,
        name="static_denominator",
    )
    history = denominator.calculate_loss_statically(detected_batch)
    assert set(history[HistoryKeys.LOSS.value]) == {
        float(detected_batch.unified_data.n_samples)
    }


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG],
    indirect=True,
)
def test_model_initialization_and_prediction(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="prediction_model",
    )
    signal_region_shift_network = model.signal_region_shift_network
    assert signal_region_shift_network is not None
    assert signal_region_shift_network.hidden.weight.dtype == torch.float64
    assert torch.all(signal_region_shift_network.hidden.bias.abs() <= 0.3)
    assert torch.all(signal_region_shift_network.output.bias.abs() <= 0.3)
    model.fit(detected_batch)
    prediction_data = detected_batch.datasets[DataSet.DataSetCategory.A_SR]
    prediction = model.predict(prediction_data)
    secondary_prediction = model.predict_secondary(prediction_data)
    theta_prediction = model.predict_theta(prediction_data)
    assert prediction.shape == (prediction_data.n_samples, 1)
    assert secondary_prediction.shape == prediction.shape
    assert torch.isfinite(torch.from_numpy(prediction)).all()
    assert torch.isfinite(torch.from_numpy(secondary_prediction)).all()
    assert torch.count_nonzero(torch.from_numpy(theta_prediction)) == 0


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG],
    indirect=True,
)
def test_checkpoint_continuation_uses_current_format(
    function_execution_context,
    data_generation,
    detector_effect,
    monkeypatch,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="checkpoint_model",
    )
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    loss = model(model._prepare_training_data(detected_batch))
    loss.backward()
    optimizer.step()

    checkpoint_path = save_training_checkpoint(
        context=function_execution_context,
        model_name="checkpoint_model",
        model=model,
        optimizer=optimizer,
        epoch=4,
        training_history={
            HistoryKeys.EPOCH.value: [4],
            HistoryKeys.LOSS.value: [1.0],
        },
    )
    checkpoint = _torch_load(checkpoint_path)
    assert checkpoint["epoch"] == 4
    assert checkpoint["optimizer_state_dict"]["state"]

    reloaded_model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="checkpoint_model",
    )
    reloaded_optimizer = reloaded_model.configure_optimizers()
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (checkpoint_path, checkpoint),
    )

    assert (
        reloaded_model._load_training_checkpoint_if_requested(reloaded_optimizer) == 5
    )
    for expected, actual in zip(model.parameters(), reloaded_model.parameters()):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITHOUT_NUISANCE_CONFIG],
    indirect=True,
)
def test_learning_rate_stays_constant_without_final_learning_rate(
    function_execution_context,
    detector_effect,
):
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="constant_learning_rate_model",
    )
    optimizer = model.configure_optimizers()

    model._set_learning_rate_for_epoch(optimizer, epoch=50)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        function_execution_context.config.train__learning_rate
    )


def test_config_rejects_final_learning_rate_above_initial_learning_rate():
    config_paths = (
        DEFAULT_CONFIG_PATHS | ONE_DIMENSION_WITH_INCREASING_LEARNING_RATE_CONFIG
    )

    with pytest.raises(
        AssertionError,
        match="Final learning rate must not exceed the initial learning rate",
    ):
        create_config_from_paths(list(config_paths.values()))


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_ADAPTIVE_LEARNING_RATE_CONFIG],
    indirect=True,
)
def test_adaptive_learning_rate_descends_over_training_epochs(
    function_execution_context,
    data_generation,
    detector_effect,
    monkeypatch,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="adaptive_learning_rate_model",
    )
    learning_rates = []

    def capture_learning_rate(optimizer, data, profiler):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        return torch.tensor(0.0)

    monkeypatch.setattr(model, "_train_step", capture_learning_rate)
    monkeypatch.setattr(
        "neural_networks.differentiating_model.save_training_checkpoint",
        lambda **_kwargs: None,
    )

    model.fit(detected_batch)

    assert learning_rates == pytest.approx([0.1, 0.07, 0.04, 0.01])


@pytest.mark.parametrize(
    "function_execution_context",
    [ONE_DIMENSION_WITH_ADAPTIVE_LEARNING_RATE_CONFIG],
    indirect=True,
)
def test_adaptive_learning_rate_continuation_uses_retargeted_epoch_schedule(
    function_execution_context,
    data_generation,
    detector_effect,
    monkeypatch,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="continued_adaptive_learning_rate_model",
    )
    learning_rates = []

    def capture_learning_rate(optimizer, data, profiler):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        return torch.tensor(0.0)

    monkeypatch.setattr(model, "_train_step", capture_learning_rate)
    monkeypatch.setattr(
        model,
        "_load_training_checkpoint_if_requested",
        lambda _optimizer: 2,
    )
    monkeypatch.setattr(
        "neural_networks.differentiating_model.save_training_checkpoint",
        lambda **_kwargs: None,
    )

    model.fit(detected_batch)

    assert learning_rates == pytest.approx([0.04, 0.01])


class _TensorboardRecorder:
    def __init__(self):
        self.scalars = []
        self.histograms = []

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))

    def add_histogram(self, tag, values, global_step):
        self.histograms.append((tag, values, global_step))


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {  # basic process
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_1D_train_config_with_nuisance.json"
            ),
        },
        {  # basic without nuisance
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_1D_train_config_without_nuisance.json"
            ),
        },
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_1D_train_config_without_nuisance_like_nplm.json"
            ),
        },
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_2D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_2D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_2D_train_config_with_nuisance.json"
            ),
        },
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_2D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_2D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_2D_train_config_without_nuisance.json"
            ),
        },
    ],
    indirect=True,
)
def test_learning(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    t_a_loss = _train_numerator(
        function_execution_context,
        detected_batch,
        detector_effect,
        "test_model",
    )

    # Train should not yet converge but a value should be given
    assert t_a_loss != 0


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/profile_1D_train_config_with_nuisance.json"
            ),
        }
    ],
    indirect=True,
)
def test_training_profile_is_saved(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    _train_numerator(
        function_execution_context,
        detected_batch,
        detector_effect,
        "profiled_model",
    )

    profile_stem = "profiled_model.1D.profile"
    trace_path = (
        function_execution_context.training_outcomes_dir / f"{profile_stem}.trace.json"
    )
    summary_path = (
        function_execution_context.training_outcomes_dir / f"{profile_stem}.txt"
    )

    assert trace_path.is_file()
    summary = summary_path.read_text()
    assert "observables: 1" in summary
    assert "hostname:" in summary
    assert "effective CPUs:" in summary
    assert "CPU affinity:" in summary
    assert "PyTorch intra-op threads:" in summary
    assert "training/forward_and_loss" in summary
    assert "training/nuisance_theta" in summary
    assert "training/backward" in summary
    assert "aten::nonzero" not in summary


def test_t_history_is_logged_to_tensorboard():
    recorder = _TensorboardRecorder()
    history = {
        HistoryKeys.EPOCH.value: [7, 9],
        HistoryKeys.NUMERATOR.value: [1.5, 1.25],
        HistoryKeys.DENOMINATOR.value: [2.0, 1.75],
        HistoryKeys.T.value: [1.0, 1.0],
    }

    log_t_history(recorder, "A", history)

    assert recorder.scalars == [
        ("A/numerator", 1.5, 7),
        ("A/numerator", 1.25, 9),
        ("A/denominator", 2.0, 7),
        ("A/denominator", 1.75, 9),
        ("A/t", 1.0, 7),
        ("A/t", 1.0, 9),
    ]
    assert recorder.histograms == []


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/long_1D_train_config_without_nuisance.json"
            ),
        },
        {
            ConfigType.DATASET.value: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/long_1D_train_config_with_nuisance.json"
            ),
        },
    ],
    indirect=True,
)
@pytest.mark.long
def test_convergence(
    function_execution_context,
    data_generation,
    detector_effect,
):
    detected_batch = detector_effect.affect_batch(data_generation.get_batch())
    t_a = _train_numerator(
        function_execution_context,
        detected_batch,
        detector_effect,
        "test_model_A",
    )

    assert t_a > 0
