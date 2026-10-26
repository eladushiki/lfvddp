from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test.environment import ConfigType
from train.checkpoints import _torch_load, checkpoint_filename, save_training_checkpoint
from train.hybrid_optimizer import HybridOptimizer, adam_warmup_epochs
from train.model_trainer import TrainLauncher


@pytest.mark.parametrize(
    "budget,warmup",
    [(1, 0), (19, 0), (20, 1), (20000, 1000), (40000, 2000), (500000, 2000)],
)
def test_warmup_boundary(budget, warmup):
    assert adam_warmup_epochs(budget) == warmup


@pytest.mark.parametrize(
    "budget,epochs,fraction,expected",
    [
        (40000, None, 0.5, 20000),
        (40000, 30000, None, 30000),
        (19, None, 0.5, 9),
        (40, 0, None, 0),
        (40, 40, None, 40),
        (40, None, 0, 0),
        (40, None, 1, 40),
    ],
)
def test_explicit_warmup_boundary(budget, epochs, fraction, expected):
    assert adam_warmup_epochs(budget, epochs, fraction) == expected


@pytest.mark.parametrize("adam_fraction", [None, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("budget", [1, 2, 20, 40])
@pytest.mark.parametrize("steps_before_save", [1, 2, 3])
def test_strict_budget_and_transactional_resume(
    budget, steps_before_save, adam_fraction
):
    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    schedule = HybridOptimizer([parameter], 0.01, budget, adam_fraction=adam_fraction)
    calls = []

    def evaluate():
        calls.append(schedule.evaluations)
        schedule.zero_grad(set_to_none=True)
        loss = (parameter.exp() - 1).square().sum()
        loss.backward()
        return loss

    for _ in range(steps_before_save):
        if schedule.evaluations < budget:
            schedule.step(evaluate)
    saved_schedule = deepcopy(schedule.schedule_state_dict())
    saved_optimizer = deepcopy(schedule.state_dict())
    saved_parameter = parameter.detach().clone()
    while schedule.evaluations < budget:
        schedule.step(evaluate)
    result = parameter.detach().clone()
    assert calls == list(range(budget))
    with torch.no_grad():
        parameter.copy_(saved_parameter)
    resumed = HybridOptimizer([parameter], 0.01, budget, adam_fraction=adam_fraction)
    resumed.load_state_dict(saved_optimizer)
    schedule = resumed
    while resumed.evaluations < budget:
        resumed.step(evaluate)
    torch.testing.assert_close(parameter, result, rtol=0, atol=0)
    extended = HybridOptimizer(
        [parameter], 0.01, budget + 100, adam_fraction=adam_fraction
    )
    extended.restore(saved_schedule)
    assert extended.warmup_epochs == saved_schedule["warmup_epochs"]
    assert extended.phase == saved_schedule["phase"]
    if budget > 1:
        with pytest.raises(ValueError, match="cannot reduce"):
            HybridOptimizer(
                [parameter], 0.01, budget - 1, adam_fraction=adam_fraction
            ).restore(saved_schedule)


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN: Path(f"test/configs/train/{fixture}.json"),
        }
        for fixture in [
            "adam_lbfgs_neural_nuisance",
            "adam_lbfgs_neural_nuisance_half_epochs",
            "adam_lbfgs_neural_nuisance_half_fraction",
            "adam_lbfgs_weight_decay_zero",
            "adam_lbfgs_weight_decay_custom",
        ]
    ],
    indirect=True,
)
@pytest.mark.parametrize("is_numerator", [False, True])
@pytest.mark.parametrize("stop_checkpoint", [1, 2, 3])
def test_hybrid_fit_checkpoint_and_prediction(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
    monkeypatch,
    is_numerator,
    stop_checkpoint,
):
    context = function_execution_context
    data = isolated_data_generation.get_batch()
    torch.manual_seed(71)
    model = differentiating_model_factory(
        context, detector_effect, is_numerator=is_numerator
    )
    model.fit(data)
    checkpoint_path = context.training_outcomes_dir / checkpoint_filename("test_model")
    checkpoint = _torch_load(checkpoint_path)
    assert checkpoint["epoch"] == context.config.train__epochs - 1
    assert checkpoint["optimizer_schedule"]["phase"] == "lbfgs"
    assert (
        checkpoint["optimizer_schedule"]["evaluations"] == context.config.train__epochs
    )
    assert model._epochs_executed == context.config.train__epochs
    assert model.minimum_loss == min(model._training_history["loss"])
    with torch.no_grad():
        assert float(model(model._prepare_training_data(data))) == pytest.approx(
            model.minimum_loss
        )
    launcher = SimpleNamespace(_config=context.config)
    assert TrainLauncher._checkpoint_finished_training(launcher, checkpoint)
    invalid = deepcopy(checkpoint)
    invalid["optimizer_schedule"]["strategy"] = "adam"
    with pytest.raises(ValueError, match="strategy differs"):
        TrainLauncher._checkpoint_finished_training(launcher, invalid)
    invalid = deepcopy(checkpoint)
    invalid["optimizer_schedule"]["total_epochs"] += 1
    with pytest.raises(ValueError, match="cannot reduce"):
        TrainLauncher._checkpoint_finished_training(launcher, invalid)
    if (
        context.config.train__adam_warmup_epochs is not None
        or context.config.train__adam_warmup_fraction is not None
    ):
        invalid = deepcopy(checkpoint)
        invalid["optimizer_schedule"]["warmup_epochs"] += 1
        with pytest.raises(ValueError, match="split differs"):
            TrainLauncher._checkpoint_finished_training(launcher, invalid)
    completed = checkpoint
    torch.manual_seed(71)
    interrupted = differentiating_model_factory(
        context, detector_effect, is_numerator=is_numerator
    )
    saves = []

    def interrupt_save(**kwargs):
        path = save_training_checkpoint(**kwargs)
        saves.append(path)
        if len(saves) == stop_checkpoint:
            raise InterruptedError("Simulated process interruption")
        return path

    monkeypatch.setattr(
        "neural_networks.differentiating_model.save_training_checkpoint", interrupt_save
    )
    with pytest.raises(InterruptedError):
        interrupted.fit(data)
    checkpoint = _torch_load(checkpoint_path)
    assert checkpoint["epoch"] < completed["epoch"]
    monkeypatch.setattr(
        "neural_networks.differentiating_model.save_training_checkpoint",
        save_training_checkpoint,
    )
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (checkpoint_path, checkpoint),
    )
    restored = differentiating_model_factory(
        context, detector_effect, is_numerator=is_numerator
    )
    restored.fit(data)
    assert restored.minimum_loss == model.minimum_loss
    for predict in ["predict", "predict_secondary", "predict_theta"]:
        expected = getattr(model, predict)(data.unified_data)
        actual = getattr(restored, predict)(data.unified_data)
        np.testing.assert_array_equal(actual, expected)
        assert np.isfinite(actual).all()


def test_line_search_budget_rolls_back_but_keeps_evaluated_candidate():
    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    schedule = HybridOptimizer([parameter], 0.01, 2)
    evaluated = []

    def evaluate():
        loss = (parameter.exp() - parameter.new_tensor(3.1).exp()).square().sum()
        evaluated.append((float(loss.detach()), parameter.detach().clone()))
        schedule.zero_grad(set_to_none=True)
        loss.backward()
        return loss

    original = parameter.detach().clone()
    schedule.step(evaluate)
    assert schedule.evaluations == 2
    torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    assert not schedule.optimizer.state
    assert schedule.state is schedule.optimizer.state
    assert schedule.param_groups is schedule.optimizer.param_groups
    assert len(evaluated) == 2
    assert not torch.equal(evaluated[0][1], evaluated[1][1])


@pytest.mark.parametrize(
    "options,message",
    [
        ({"train__optimizer_strategy": "unknown"}, "Unknown"),
        *[
            ({"train__adam_weight_decay": value}, "finite non-negative")
            for value in [-0.1, float("nan"), float("inf"), True, "0.01", None]
        ],
        (
            {"train__optimizer_strategy": "adam_lbfgs", "train__backend": "nplm"},
            "only supported",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__final_learning_rate": 0.001,
            },
            "constant",
        ),
        ({"train__epochs": 0}, "positive"),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_epochs": 41,
            },
            "within",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_epochs": -1,
            },
            "within",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_epochs": 0.5,
            },
            "integer",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_epochs": True,
            },
            "integer",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_fraction": -0.1,
            },
            "between",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_fraction": 1.1,
            },
            "between",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_fraction": float("nan"),
            },
            "finite",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_fraction": float("inf"),
            },
            "finite",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_fraction": True,
            },
            "finite",
        ),
        (
            {
                "train__optimizer_strategy": "adam_lbfgs",
                "train__adam_warmup_epochs": 20,
                "train__adam_warmup_fraction": 0.5,
            },
            "not both",
        ),
        ({"train__adam_warmup_epochs": 20}, "require"),
    ],
)
def test_optimizer_config_validation(options, message):
    from train.train_config import TrainConfig

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


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET: Path(
                "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
            ),
            ConfigType.DETECTOR: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN: Path("test/configs/train/adam_lbfgs_no_nuisance.json"),
        }
    ],
    indirect=True,
)
def test_hybrid_static_denominator(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    differentiating_model_factory,
):
    model = differentiating_model_factory(
        function_execution_context, detector_effect, is_numerator=False
    )
    history = model.fit(isolated_data_generation.get_batch())
    assert model.minimum_loss == history["loss"][-1]
    assert model._training_history == history
    assert model._epochs_executed == 0


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("adam_fraction", [None, 0.5])
def test_hybrid_optimizer_state_round_trip(legacy, adam_fraction):
    from train.hybrid_optimizer import load_optimizer_state

    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    optimizer = HybridOptimizer([parameter], 0.01, 40, adam_fraction=adam_fraction)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter.exp() - 1).square().sum()
        loss.backward()
        return loss

    for _ in range(optimizer.warmup_epochs + 2):
        optimizer.step(closure)
    assert optimizer.phase == "lbfgs"
    assert optimizer.optimizer.state[parameter]["old_dirs"]
    saved_state = deepcopy(optimizer.state_dict())
    saved_parameter = parameter.detach().clone()
    while optimizer.evaluations < optimizer.total_epochs:
        optimizer.step(closure)
    expected = parameter.detach().clone()
    optimizer = HybridOptimizer([parameter], 0.01, 40, adam_fraction=adam_fraction)
    if legacy:
        schedule = saved_state.pop("optimizer_schedule")
        load_optimizer_state(optimizer, saved_state, schedule)
    else:
        optimizer.load_state_dict(saved_state)
    with torch.no_grad():
        parameter.copy_(saved_parameter)
    while optimizer.evaluations < optimizer.total_epochs:
        optimizer.step(closure)
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)


def test_checkpoint_keeps_large_boundary_and_rejects_changed_split():
    from train.hybrid_optimizer import validate_optimizer_continuation

    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    original = HybridOptimizer([parameter], 0.01, 40000, adam_fraction=0.5)
    saved = original.schedule_state_dict()
    restored = HybridOptimizer([parameter], 0.01, 80000, adam_fraction=0.5)
    restored.restore(saved)
    assert restored.warmup_epochs == 20000
    second_extension = HybridOptimizer([parameter], 0.01, 100000, adam_fraction=0.5)
    second_extension.restore(restored.schedule_state_dict())
    assert second_extension.warmup_epochs == 20000
    assert second_extension.initial_total_epochs == 40000
    legacy = deepcopy(saved)
    legacy.pop("initial_total_epochs")
    restored.restore(legacy)
    assert restored.initial_total_epochs == 40000
    # Both public continuation gates and direct optimizer restoration reject changes.
    for epochs, fraction in [(10000, None), (None, 0.25)]:
        with pytest.raises(ValueError, match="split differs"):
            validate_optimizer_continuation(
                saved, "adam_lbfgs", 80000, epochs, fraction
            )
        with pytest.raises(ValueError, match="split differs"):
            HybridOptimizer([parameter], 0.01, 80000, epochs, fraction).restore(saved)
    implicit = HybridOptimizer([parameter], 0.01, 80000)
    implicit.restore(saved)
    assert implicit.warmup_epochs == 20000


def test_native_optimizer_hooks_and_zero_grad():
    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    optimizer = HybridOptimizer([parameter], 0.01, 40)
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert HybridOptimizer.zero_grad is torch.optim.Optimizer.zero_grad
    assert HybridOptimizer.state_dict is torch.optim.Optimizer.state_dict
    assert HybridOptimizer.load_state_dict is torch.optim.Optimizer.load_state_dict
    parameter.grad = torch.ones_like(parameter)
    optimizer.zero_grad(set_to_none=False)
    torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))
    optimizer.zero_grad()
    assert parameter.grad is None
    events = []
    optimizer.register_step_pre_hook(
        lambda opt, args, kwargs: events.append(("step_pre", opt))
    )
    optimizer.register_step_post_hook(
        lambda opt, args, kwargs: events.append(("step_post", opt))
    )
    optimizer.register_state_dict_pre_hook(lambda opt: events.append(("save_pre", opt)))

    def saved(opt, state):
        assert "optimizer_schedule" in state
        events.append(("save_post", opt))

    optimizer.register_state_dict_post_hook(saved)
    optimizer.register_load_state_dict_pre_hook(
        lambda opt, state: events.append(("load_pre", opt))
    )

    def loaded(opt):
        assert opt.optimizer.state is opt.state
        assert opt.optimizer.param_groups is opt.param_groups
        events.append(("load_post", opt))

    optimizer.register_load_state_dict_post_hook(loaded)

    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    state = deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(state)
    assert [name for name, _ in events] == [
        "step_pre",
        "step_post",
        "save_pre",
        "save_post",
        "load_pre",
        "load_post",
    ]
    assert all(opt is optimizer for _, opt in events)


def test_scheduler_and_shared_references_across_phase_switch():
    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    optimizer = HybridOptimizer(
        [{"params": [parameter], "param_names": ["weight"]}], 0.01, 40
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    groups, group, state = (
        optimizer.param_groups,
        optimizer.param_groups[0],
        optimizer.state,
    )

    def closure():
        optimizer.zero_grad()
        loss = (parameter.exp() - 1).square().sum()
        loss.backward()
        return loss

    for _ in range(2):
        optimizer.step(closure)
        scheduler.step()
    assert optimizer.phase == "adam"
    assert group["lr"] == pytest.approx(0.0025)
    optimizer.step(closure)
    assert optimizer.phase == "lbfgs"
    assert group["lr"] == 1.0
    assert optimizer.param_groups is groups
    assert optimizer.param_groups[0] is group
    assert optimizer.state is state
    assert optimizer.optimizer.param_groups is groups
    assert optimizer.optimizer.state is state
    assert group["initial_lr"] == 0.01
    assert group["param_names"] == ["weight"]
    scheduler.step()
    assert optimizer.optimizer.param_groups[0]["lr"] == 0.5
    saved = deepcopy(optimizer.state_dict())
    optimizer.load_state_dict(saved)
    assert scheduler.optimizer is optimizer
    assert optimizer.optimizer.param_groups is optimizer.param_groups
    assert optimizer.optimizer.state is optimizer.state
    scheduler.step()
    assert optimizer.optimizer.param_groups[0]["lr"] == 0.25


def test_hybrid_rejects_multiple_parameter_groups():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    with pytest.raises(ValueError, match="one parameter group"):
        HybridOptimizer([{"params": [first]}, {"params": [second]}], 0.01, 40)
    optimizer = HybridOptimizer([first], 0.01, 40)
    with pytest.raises(ValueError, match="one parameter group"):
        optimizer.add_param_group({"params": [second]})
    assert optimizer.param_groups[0]["params"] == [first]
