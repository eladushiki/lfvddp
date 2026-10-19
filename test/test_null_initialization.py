from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from test.environment import ConfigType
from train.checkpoints import _torch_load, checkpoint_filename, save_training_checkpoint
from train.model_trainer import ParallelTrainLauncher, SequentialTrainLauncher
from train.null_initialization import (
    prepare_null_normalization,
    retain_null_candidate,
    validate_null_initialization_checkpoint,
)
from train.runtime_resources import RuntimeAllocation
from train.single_train import select_train_launcher_class
from train.train_config import TrainConfig

CASES = [
    {
        ConfigType.DATASET: Path(
            f"test/configs/dataset/disjoint_{dimension}D_generated_dataset_config.json"
        ),
        ConfigType.DETECTOR: Path(
            f"test/configs/detector/basic_{dimension}D_detector_config.json"
        ),
        ConfigType.TRAIN: Path(f"test/configs/train/{name}.json"),
    }
    for dimension, name in [
        (1, "null_initialized_neural"),
        (1, "null_initialized_static"),
        (2, "null_initialized_binned_2d"),
        (1, "null_initialized_adam"),
    ]
]
ALLOCATION = RuntimeAllocation(
    cpu_count=2,
    cpu_affinity=(),
    assigned_gpu_ids=(),
    visible_gpu_count=0,
    gpu_names=(),
    gpu_total_memory_bytes=(),
)


@pytest.mark.parametrize("function_execution_context", CASES, indirect=True)
def test_exact_null_embedding_and_learning(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    data = isolated_data_generation.get_batch()
    null = differentiating_model_factory(
        function_execution_context, detector_effect, is_numerator=False, name="null"
    )
    null.fit(data)
    alternative = differentiating_model_factory(
        function_execution_context, detector_effect, name="alternative"
    )
    prepare_null_normalization(alternative, null)
    prepared = alternative._prepare_training_data(data)
    signal = alternative.signal_region_shift_network
    hidden = [
        p.detach().clone() for layer in signal.hidden_layers for p in layer.parameters()
    ]
    assert retain_null_candidate(alternative, null, prepared, initialize=True)
    assert alternative._norm_factor is not null._norm_factor
    assert alternative._norm_factor.to_mapping() == null._norm_factor.to_mapping()
    assert alternative._best_epoch == -1
    assert alternative._training_history == {}
    assert alternative._epochs_executed == 0
    assert float(alternative(prepared).detach()) == float(null(prepared).detach())
    assert alternative.minimum_loss == null.minimum_loss
    for expected, actual in zip(
        hidden, (p for layer in signal.hidden_layers for p in layer.parameters())
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if null.nuisance_function_space is not None:
        for a, b in zip(
            null.nuisance_function_space.parameters(),
            alternative.nuisance_function_space.parameters(),
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert a.data_ptr() != b.data_ptr()
    optimizer = torch.optim.Adam(alternative.parameters(), lr=0.01)
    optimizer.zero_grad()
    alternative(prepared).backward()
    assert signal.output.weight.grad.abs().max() > 0
    assert all(
        torch.count_nonzero(p.grad) == 0
        for layer in signal.hidden_layers
        for p in layer.parameters()
    )
    optimizer.step()
    optimizer.zero_grad()
    alternative(prepared).backward()
    assert any(
        torch.count_nonzero(p.grad) > 0
        for layer in signal.hidden_layers
        for p in layer.parameters()
    )
    optimizer.step()
    assert any(
        not torch.equal(a, b)
        for a, b in zip(
            hidden, (p for layer in signal.hidden_layers for p in layer.parameters())
        )
    )


@pytest.mark.parametrize("function_execution_context", CASES, indirect=True)
def test_dependent_launcher_budget_and_predictions(
    function_execution_context, isolated_data_generation, detector_effect, monkeypatch
):
    context = function_execution_context
    data = isolated_data_generation.get_batch()
    assert (
        select_train_launcher_class(context.config, ALLOCATION)
        is SequentialTrainLauncher
    )
    launcher = SequentialTrainLauncher(context, detector_effect, allocation=ALLOCATION)
    numerator = launcher.add_training(data, detector_effect, True, name="alternative")
    denominator = launcher.add_training(data, detector_effect, False, name="null")
    order = []
    original = launcher._execute_in_parent

    def execute(assignment):
        order.append(launcher.get_training(assignment.index).is_numerator)
        original(assignment)

    monkeypatch.setattr(launcher, "_execute_in_parent", execute)
    launcher.execute_trainings()
    assert order == [False, True]
    alternative, null = (
        launcher.get_training(numerator),
        launcher.get_training(denominator),
    )
    assert alternative.result <= null.result
    assert alternative.model._epochs_executed == context.config.train__epochs
    assert null.model._epochs_executed == (
        0 if context.config.train__nuisance is None else context.config.train__epochs
    )
    assert len(alternative.history["epoch"]) == context.config.train__epochs
    assert alternative.history["epoch"][0] == 0
    assert -1 not in alternative.history["epoch"]
    assert not any(
        name.startswith("null_model")
        for name, _ in alternative.model.named_parameters()
    )
    for predict in (
        alternative.model.predict,
        alternative.model.predict_secondary,
        alternative.model.predict_theta,
    ):
        predictions = predict(data.unified_data)
        assert predictions.shape == (data.unified_data.n_samples, 1)
        assert np.isfinite(predictions).all()
    with pytest.raises(RuntimeError, match="Sequential"):
        ParallelTrainLauncher(
            context, detector_effect, allocation=ALLOCATION
        ).execute_trainings()


@pytest.mark.parametrize("function_execution_context", [CASES[0]], indirect=True)
@pytest.mark.parametrize("stop_checkpoint", [1, 2, 3, 40])
def test_partial_and_completed_pair_resume(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    monkeypatch,
    stop_checkpoint,
):
    context = function_execution_context
    data = isolated_data_generation.get_batch()
    launcher = SequentialTrainLauncher(context, detector_effect, allocation=ALLOCATION)
    launcher.add_training(data, detector_effect, True, name="alternative")
    launcher.add_training(data, detector_effect, False, name="null")
    launcher.execute_trainings()
    expected = launcher.get_training(0).model.predict(data.unified_data)
    expected_history = deepcopy(launcher.get_training(0).history)
    interrupted = SequentialTrainLauncher(
        context, detector_effect, allocation=ALLOCATION
    )
    interrupted.add_training(data, detector_effect, True, name="alternative")
    interrupted.add_training(data, detector_effect, False, name="null")
    calls = []

    def save(**kwargs):
        path = save_training_checkpoint(**kwargs)
        if kwargs["model_name"] == "alternative":
            calls.append(path)
            if (
                len(calls) == stop_checkpoint
                or kwargs["epoch"] == context.config.train__epochs - 1
            ):
                raise InterruptedError
        return path

    monkeypatch.setattr(
        "neural_networks.differentiating_model.save_training_checkpoint", save
    )
    with pytest.raises(InterruptedError):
        interrupted.execute_trainings()
    saved = _torch_load(
        context.training_outcomes_dir / checkpoint_filename("alternative")
    )
    assert saved["initialize_from_null"] is True

    def find(_context, name, **kwargs):
        path = context.training_outcomes_dir / checkpoint_filename(name)
        return (path, _torch_load(path)) if path.exists() else None

    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint", find
    )
    monkeypatch.setattr(
        "neural_networks.differentiating_model.save_training_checkpoint",
        save_training_checkpoint,
    )
    resumed = SequentialTrainLauncher(context, detector_effect, allocation=ALLOCATION)
    resumed.add_training(data, detector_effect, True, name="alternative")
    resumed.add_training(data, detector_effect, False, name="null")
    resumed.execute_trainings()
    actual = resumed.get_training(0)
    np.testing.assert_array_equal(actual.model.predict(data.unified_data), expected)
    assert actual.history == expected_history
    assert (
        actual.model._epochs_executed
        == context.config.train__epochs - saved["epoch"] - 1
    )
    assert resumed.get_training(1).model is not None
    assert resumed.get_training(1).model._epochs_executed == 0


@pytest.mark.parametrize(
    "options,message",
    [
        ({"train__initialize_from_null": "true"}, "boolean"),
        ({"train__initialize_from_null": 1}, "boolean"),
        (
            {"train__initialize_from_null": True, "train__backend": "nplm"},
            "requires LFVDDP",
        ),
        (
            {
                "train__initialize_from_null": True,
                "train__f": {
                    "family": "bin_indicators",
                    "options": {"minima": [0], "maxima": [10], "number_of_bins": [10]},
                },
            },
            "requires LFVDDP",
        ),
    ],
)
def test_null_initialization_config_validation(options, message):
    with pytest.raises(ValueError, match=message):
        TrainConfig(
            **(
                {
                    "train__epochs": 40,
                    "train__number_of_epochs_for_checkpoint": 1,
                    "train__f": {
                        "family": "adaptive_neural",
                        "options": {"hidden_layer_nodes": 4},
                    },
                }
                | options
            )
        )


def test_checkpoint_initialization_policy():
    validate_null_initialization_checkpoint({}, False)
    for saved, requested in [({}, True), ({"initialize_from_null": True}, False)]:
        with pytest.raises(ValueError, match="differs"):
            validate_null_initialization_checkpoint(saved, requested)


@pytest.mark.parametrize("function_execution_context", [CASES[0]], indirect=True)
def test_requires_explicit_fitted_null_and_unambiguous_pair(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    from neural_networks.differentiating_model import calc_min_LFVNN

    context = function_execution_context
    data = isolated_data_generation.get_batch()
    alternative = differentiating_model_factory(context, detector_effect)
    with pytest.raises(ValueError, match="fitted denominator"):
        alternative.fit(data)
    with pytest.raises(ValueError, match="fitted denominator"):
        calc_min_LFVNN(context, data, detector_effect, True, "missing_null")
    null = differentiating_model_factory(context, detector_effect, is_numerator=False)
    null._prepare_training_data(data)
    with pytest.raises(ValueError, match="learned denominator must be fitted"):
        prepare_null_normalization(alternative, null)
    launcher = SequentialTrainLauncher(context, detector_effect, allocation=ALLOCATION)
    launcher.add_training(data, detector_effect, True)
    with pytest.raises(ValueError, match="one denominator and one alternative"):
        launcher.execute_trainings()
    launcher.add_training(data, detector_effect, False)
    launcher.add_training(data, detector_effect, False)
    with pytest.raises(ValueError, match="one denominator and one alternative"):
        launcher.execute_trainings()


@pytest.mark.parametrize("function_execution_context", [CASES[0]], indirect=True)
@pytest.mark.parametrize(
    "training_context_pair",
    [("null_initialized_one_epoch_adam.json", "null_initialized_extended_adam.json")],
    indirect=True,
)
def test_completed_alternative_retains_new_null_without_reset(
    training_context_pair,
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    monkeypatch,
    differentiating_model_factory,
):
    from data_tools.detector.detector_effect import DetectorEffect

    short, extended = training_context_pair
    data = isolated_data_generation.get_batch()
    short_detector, long_detector = DetectorEffect(short), DetectorEffect(extended)
    launcher = SequentialTrainLauncher(short, short_detector, allocation=ALLOCATION)
    launcher.add_training(data, short_detector, True, name="alternative")
    launcher.add_training(data, short_detector, False, name="null")
    launcher.execute_trainings()
    old_null = launcher.get_training(1).result
    old_path = short.training_outcomes_dir / checkpoint_filename("alternative")
    before = _torch_load(old_path)
    assert before["best_epoch"] == -1
    null_path = short.training_outcomes_dir / checkpoint_filename("null")
    null_checkpoint = _torch_load(null_path)
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (null_path, null_checkpoint),
    )
    null = differentiating_model_factory(
        extended, long_detector, is_numerator=False, name="null"
    )
    null.fit(data)
    assert null.minimum_loss < old_null
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (old_path, before),
    )
    alternative = differentiating_model_factory(
        short, short_detector, name="alternative"
    )
    alternative.fit(data, null_model=null)
    assert alternative._epochs_executed == 0
    assert alternative.minimum_loss == null.minimum_loss
    assert alternative.minimum_loss < before["best_loss"]
    after = _torch_load(old_path)
    assert after["epoch"] == before["epoch"]
    assert after["training_history"] == before["training_history"]
    assert after["best_loss"] == null.minimum_loss
    assert (
        after["optimizer_state_dict"]["param_groups"]
        == before["optimizer_state_dict"]["param_groups"]
    )
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (old_path, after),
    )
    again = differentiating_model_factory(short, short_detector, name="alternative")
    again.fit(data, null_model=null)
    assert again.minimum_loss == alternative.minimum_loss
    np.testing.assert_array_equal(
        again.predict(data.unified_data), alternative.predict(data.unified_data)
    )
    for key in before["model_state_dict"]:
        torch.testing.assert_close(
            after["model_state_dict"][key],
            before["model_state_dict"][key],
            rtol=0,
            atol=0,
        )
    for param, state in before["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                after["optimizer_state_dict"]["state"][param][key],
                value,
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("function_execution_context", [CASES[0]], indirect=True)
@pytest.mark.parametrize(
    "training_context_pair",
    [("null_initialized_neural.json", "null_initialized_neural_extended.json")],
    indirect=True,
)
def test_pair_budget_extension_preserves_original_warmup(
    training_context_pair,
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    monkeypatch,
):
    from data_tools.detector.detector_effect import DetectorEffect

    short, extended = training_context_pair
    data = isolated_data_generation.get_batch()
    first = SequentialTrainLauncher(short, DetectorEffect(short), allocation=ALLOCATION)
    first.add_training(data, first._detector_effect, True, name="alternative")
    first.add_training(data, first._detector_effect, False, name="null")
    first.execute_trainings()
    saved = {
        name: (
            short.training_outcomes_dir / checkpoint_filename(name),
            _torch_load(short.training_outcomes_dir / checkpoint_filename(name)),
        )
        for name in ("alternative", "null")
    }
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda _context, name, **kwargs: saved[name],
    )
    second = SequentialTrainLauncher(
        extended, DetectorEffect(extended), allocation=ALLOCATION
    )
    second.add_training(data, second._detector_effect, True, name="alternative")
    second.add_training(data, second._detector_effect, False, name="null")
    second.execute_trainings()
    assert second.get_training(0).result <= second.get_training(1).result
    for index, name in enumerate(("alternative", "null")):
        record = second.get_training(index)
        assert record.model._epochs_executed == 20
        assert record.history["loss"][:40] == first.get_training(index).history["loss"]
        final = _torch_load(extended.training_outcomes_dir / checkpoint_filename(name))
        assert final["optimizer_schedule"]["warmup_epochs"] == 2
        assert final["optimizer_schedule"]["evaluations"] == 60


@pytest.mark.parametrize(
    "function_execution_context", [CASES[0], CASES[2]], indirect=True
)
def test_null_initialized_public_training_plot(
    function_execution_context, isolated_data_generation, detector_effect
):
    from train.single_train import train_for_t

    statistic = train_for_t(
        function_execution_context,
        isolated_data_generation.get_batch(),
        detector_effect,
        "initialized",
        ALLOCATION,
    )
    assert np.isfinite(statistic)
    assert statistic >= 0
    assert list(
        function_execution_context.unique_out_dir.glob("dataset_process_plot*.png")
    )


@pytest.mark.parametrize("function_execution_context", [CASES[0]], indirect=True)
def test_legacy_flag_rejected_before_loading_state(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
    monkeypatch,
):
    context = function_execution_context
    data = isolated_data_generation.get_batch()
    launcher = SequentialTrainLauncher(context, detector_effect, allocation=ALLOCATION)
    launcher.add_training(data, detector_effect, True, name="alternative")
    launcher.add_training(data, detector_effect, False, name="null")
    launcher.execute_trainings()
    path = context.training_outcomes_dir / checkpoint_filename("alternative")
    legacy = _torch_load(path)
    legacy.pop("initialize_from_null")
    with pytest.raises(ValueError, match="differs"):
        launcher._checkpoint_finished_training(legacy)
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (path, legacy),
    )
    model = differentiating_model_factory(context, detector_effect, name="alternative")
    with pytest.raises(ValueError, match="differs"):
        model.fit(data, null_model=launcher.get_training(1).model)


@pytest.mark.parametrize("function_execution_context", [CASES[0]], indirect=True)
def test_mismatched_normalization_is_rejected(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    from data_tools.data_utils import ShiftAndNormalizationFactor

    context = function_execution_context
    data = isolated_data_generation.get_batch()
    null = differentiating_model_factory(context, detector_effect, is_numerator=False)
    null.fit(data)
    alternative = differentiating_model_factory(context, detector_effect)
    mapping = null._norm_factor.to_mapping()
    observable = next(iter(mapping["offsets"]))
    mapping["offsets"][observable] += 1
    alternative._norm_factor = ShiftAndNormalizationFactor.from_mapping(mapping)
    with pytest.raises(ValueError, match="normalization maps differ"):
        alternative.fit(data, null_model=null)


@pytest.mark.parametrize("function_execution_context", [CASES[0]], indirect=True)
def test_initialization_uses_retained_best_null(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    context = function_execution_context
    data = isolated_data_generation.get_batch()
    null = differentiating_model_factory(context, detector_effect, is_numerator=False)
    null.fit(data)
    with torch.no_grad():
        for parameter in null.parameters():
            parameter.add_(1.0)
    alternative = differentiating_model_factory(context, detector_effect)
    prepare_null_normalization(alternative, null)
    prepared = alternative._prepare_training_data(data)
    retain_null_candidate(alternative, null, prepared, initialize=True)
    assert alternative.minimum_loss == null.minimum_loss
