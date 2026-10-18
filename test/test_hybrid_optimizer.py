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


@pytest.mark.parametrize("budget", [1, 2, 20, 40])
@pytest.mark.parametrize("steps_before_save", [1, 2, 3])
def test_strict_budget_and_transactional_resume(budget, steps_before_save):
    parameter = torch.nn.Parameter(torch.tensor([3.0], dtype=torch.float64))
    schedule = HybridOptimizer([parameter], 0.01, budget)
    calls = []

    def evaluate(epoch):
        calls.append(epoch)
        return (parameter.exp() - 1).square().sum()

    for _ in range(steps_before_save):
        if schedule.evaluations < budget:
            schedule.step(evaluate)
    saved_schedule = deepcopy(schedule.state_dict())
    saved_optimizer = deepcopy(schedule.optimizer.state_dict())
    saved_parameter = parameter.detach().clone()
    while schedule.evaluations < budget:
        schedule.step(evaluate)
    result = parameter.detach().clone()
    assert calls == list(range(budget))
    with torch.no_grad():
        parameter.copy_(saved_parameter)
    resumed = HybridOptimizer([parameter], 0.01, budget)
    resumed.restore(saved_schedule)
    resumed.optimizer.load_state_dict(saved_optimizer)
    while resumed.evaluations < budget:
        resumed.step(evaluate)
    torch.testing.assert_close(parameter, result, rtol=0, atol=0)
    extended = HybridOptimizer([parameter], 0.01, budget + 100)
    extended.restore(saved_schedule)
    assert extended.warmup_epochs == saved_schedule["warmup_epochs"]
    assert extended.phase == saved_schedule["phase"]
    if budget > 1:
        with pytest.raises(ValueError, match="cannot reduce"):
            HybridOptimizer([parameter], 0.01, budget - 1).restore(saved_schedule)


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
            ConfigType.TRAIN: Path(
                "test/configs/train/adam_lbfgs_neural_nuisance.json"
            ),
        }
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

    def evaluate(epoch):
        loss = (parameter.exp() - parameter.new_tensor(3.1).exp()).square().sum()
        evaluated.append((float(loss.detach()), parameter.detach().clone()))
        return loss

    original = parameter.detach().clone()
    schedule.step(evaluate)
    assert schedule.evaluations == 2
    torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    assert not schedule.optimizer.state
    assert len(evaluated) == 2
    assert not torch.equal(evaluated[0][1], evaluated[1][1])


@pytest.mark.parametrize(
    "options,message",
    [
        ({"train__optimizer_strategy": "unknown"}, "Unknown"),
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
