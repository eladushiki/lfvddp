"""Public regression coverage for the canonical adaptive path and nuisance modes."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_tools.data_utils import DataSet
from neural_networks.differentiating_model import DifferentiatingModel
from test.environment import ConfigType
from train.checkpoint_metadata import build_checkpoint_metadata
from train.checkpoints import _torch_load, save_training_checkpoint


_DATASET = Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json")
_DETECTOR = Path("test/configs/detector/basic_1D_detector_config.json")
_TRAIN = Path("test/configs/train")


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            {ConfigType.DATASET: _DATASET, ConfigType.DETECTOR: _DETECTOR, ConfigType.TRAIN: _TRAIN / "baseline_1D_omitted_f_disabled_nuisance.json"},
            id="adaptive-disabled-nuisance",
        ),
        pytest.param(
            {ConfigType.DATASET: _DATASET, ConfigType.DETECTOR: _DETECTOR, ConfigType.TRAIN: _TRAIN / "baseline_1D_omitted_f_neural_nuisance.json"},
            id="adaptive-neural-nuisance",
        ),
        pytest.param(
            {ConfigType.DATASET: _DATASET, ConfigType.DETECTOR: _DETECTOR, ConfigType.TRAIN: _TRAIN / "baseline_1D_omitted_f_binned_nuisance.json"},
            id="adaptive-binned-nuisance",
        ),
    ],
    indirect=True,
)
def test_canonical_adaptive_model_public_contract(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    tmp_path,
):
    batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    prediction_data = batch.datasets[DataSet.DataSetCategory.A_SR]
    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="adaptive_baseline",
        dtype=torch.float64,
    )
    prepared = model._prepare_training_data(batch)
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    loss = model(prepared)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    predictions = (
        model.predict(prediction_data),
        model.predict_secondary(prediction_data),
        model.predict_theta(prediction_data),
    )
    assert all(prediction.shape == (prediction_data.n_samples, 1) for prediction in predictions)
    assert all(np.isfinite(prediction).all() for prediction in predictions)

    checkpoint_path = save_training_checkpoint(
        context=SimpleNamespace(
            training_outcomes_dir=tmp_path,
            array_index=function_execution_context.array_index,
            run_hash=function_execution_context.run_hash,
        ),
        model_name="adaptive_baseline",
        model=model,
        optimizer=optimizer,
        epoch=0,
        training_history=model._training_history,
        metadata=build_checkpoint_metadata(
            model_name="adaptive_baseline",
            is_numerator=True,
            resolved_config=model._function_space_config,
            normalization_factor=model._norm_factor,
        ),
    )
    restored = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="adaptive_baseline",
        dtype=torch.float64,
    )
    restored._norm_factor = model._norm_factor
    restored._prepare_training_data(batch)
    restored.load_state_dict(_torch_load(checkpoint_path)["model_state_dict"], strict=True)
    np.testing.assert_allclose(restored.predict(prediction_data), predictions[0])
