from copy import deepcopy
from pathlib import Path

import pytest
import torch

from test.environment import ConfigType
from train.hybrid_optimizer import HybridOptimizer


@pytest.mark.parametrize(
    "function_execution_context,expected",
    [
        (
            {
                ConfigType.TRAIN: Path(f"test/configs/train/{fixture}.json"),
                ConfigType.DATASET: Path(
                    "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
                ),
                ConfigType.DETECTOR: Path(
                    "test/configs/detector/basic_1D_detector_config.json"
                ),
            },
            decay,
        )
        for fixture, decay in [
            ("short_1D_train_config_with_neural_nuisance", 1e-4),
            ("adam_weight_decay_zero", 0),
            ("adam_weight_decay_custom", 0.02),
            ("adam_lbfgs_neural_nuisance", 1e-4),
            ("adam_lbfgs_weight_decay_zero", 0),
            ("adam_lbfgs_weight_decay_custom", 0.02),
        ]
    ],
    indirect=["function_execution_context"],
)
@pytest.mark.parametrize("is_numerator", [False, True])
def test_adam_decay_configuration_and_restore(
    function_execution_context,
    detector_effect,
    isolated_data_generation,
    differentiating_model_factory,
    expected,
    is_numerator,
):
    model = differentiating_model_factory(
        function_execution_context, detector_effect, is_numerator=is_numerator
    )
    model.fit(isolated_data_generation.get_batch())
    optimizer = model.configure_optimizers()
    assert function_execution_context.config.train__adam_weight_decay == expected
    assert optimizer.param_groups[0]["weight_decay"] == expected
    saved = deepcopy(optimizer.state_dict())

    # With zero data gradients, only decay can move the parameters.
    parameters = optimizer.param_groups[0]["params"]
    before = [p.detach().clone() for p in parameters]

    def closure():
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        return parameters[0].new_zeros(())

    optimizer.step(closure)
    assert any(not torch.equal(p, old) for p, old in zip(parameters, before)) == (
        expected > 0
    )
    restored = model.configure_optimizers()
    restored.param_groups[0]["weight_decay"] = expected + 1
    restored.load_state_dict(saved)
    assert restored.param_groups[0]["weight_decay"] == expected
    if isinstance(restored, HybridOptimizer):
        for _ in range(restored.warmup_epochs):
            restored.step(closure)
        restored.step(closure)
        assert restored.phase == "lbfgs"
        assert "weight_decay" not in restored.param_groups[0]
