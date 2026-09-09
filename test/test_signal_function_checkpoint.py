"""File-backed checkpoint and continuation compatibility for Issue 018 function spaces."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_tools.data_utils import DataSet
from frame.file_system.training_history import HistoryKeys
from neural_networks.differentiating_model import DifferentiatingModel
from test.environment import ConfigType
from train.checkpoints import (
    _torch_load,
    checkpoint_metadata_path,
    load_checkpoint_metadata,
    save_training_checkpoint,
)


_DATASET = Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json")
_DETECTOR = Path("test/configs/detector/basic_1D_detector_config.json")

# One file-backed fixture per supported trainable f family.  Nuisance remains
# enabled in every case so the checkpoint covers the complete model state.
_CASES = [
    pytest.param(
        {
            ConfigType.DATASET: _DATASET,
            ConfigType.DETECTOR: _DETECTOR,
            ConfigType.TRAIN: Path("test/configs/train/issue018_s04_1D_adaptive_neural.json"),
        },
        id="adaptive-neural",
    ),
    pytest.param(
        {
            ConfigType.DATASET: _DATASET,
            ConfigType.DETECTOR: _DETECTOR,
            ConfigType.TRAIN: Path("test/configs/train/issue018_s04_1D_cubic_binned.json"),
        },
        id="cubic-bspline",
    ),
    pytest.param(
        {
            ConfigType.DATASET: _DATASET,
            ConfigType.DETECTOR: _DETECTOR,
            ConfigType.TRAIN: Path("test/configs/train/issue018_s04_1D_legendre_binned.json"),
        },
        id="legendre-polynomial",
    ),
    pytest.param(
        {
            ConfigType.DATASET: _DATASET,
            ConfigType.DETECTOR: _DETECTOR,
            ConfigType.TRAIN: Path("test/configs/train/issue018_s04_1D_gaussian_binned.json"),
        },
        id="gaussian-rbf",
    ),
    pytest.param(
        {
            ConfigType.DATASET: _DATASET,
            ConfigType.DETECTOR: _DETECTOR,
            ConfigType.TRAIN: Path("test/configs/train/issue018_s04_1D_sigmoid_binned.json"),
        },
        id="fixed-sigmoid",
    ),
]

_CONTINUATION_CONFIG = {
    ConfigType.DATASET: _DATASET,
    ConfigType.DETECTOR: _DETECTOR,
    ConfigType.TRAIN: Path(
        "test/configs/train/issue018_s04_1D_adaptive_neural_continuation.json"
    ),
}


def _checkpoint_context(tmp_path, context):
    return SimpleNamespace(
        training_outcomes_dir=tmp_path,
        array_index=context.array_index,
        run_hash=context.run_hash,
    )


def _make_model(context, detector_effect, name):
    return DifferentiatingModel(
        context=context,
        detector_effect=detector_effect,
        is_numerator=True,
        name=name,
        dtype=torch.float64,
        device="cpu",
    )


def _take_one_step(model, data_batch):
    prepared = model._prepare_training_data(data_batch)
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    loss = model(prepared)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.nuisance_calculation.clamp_parameters()
    model._log(0, loss)
    return optimizer


@pytest.mark.parametrize("function_execution_context", _CASES, indirect=True)
def test_checkpoint_round_trip_preserves_all_function_space_state(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    tmp_path,
):
    context = function_execution_context
    data_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = _make_model(context, detector_effect, "checkpoint_round_trip")
    optimizer = _take_one_step(model, data_batch)

    checkpoint_path = save_training_checkpoint(
        context=_checkpoint_context(tmp_path, context),
        model_name="checkpoint_round_trip",
        model=model,
        optimizer=optimizer,
        epoch=0,
        training_history=model._training_history,
    )
    checkpoint = _torch_load(checkpoint_path)

    # The legacy payload shape is part of the baseline contract.  Metadata is
    # deliberately stored beside it rather than adding a top-level key.
    assert set(checkpoint) == {
        "model_name",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "training_history",
        "array_index",
        "run_hash",
    }
    assert checkpoint["epoch"] == 0
    assert checkpoint["training_history"][HistoryKeys.EPOCH.value] == [0]
    assert checkpoint["optimizer_state_dict"]["state"]
    assert tuple(checkpoint["model_state_dict"]) == tuple(model.state_dict())

    metadata = load_checkpoint_metadata(checkpoint_path)
    assert metadata is not None
    assert metadata["config_fingerprint"] == model.checkpoint_metadata()["config_fingerprint"]
    assert metadata["normalization_factor"]["factors"] == {
        name: model._norm_factor.get_factor(name)
        for name in model._norm_factor._factors
    }
    assert checkpoint_metadata_path(checkpoint_path).exists()

    restored = _make_model(context, detector_effect, "checkpoint_round_trip")
    incompatible = restored.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert tuple(restored.state_dict()) == tuple(model.state_dict())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)

    restored_optimizer = restored.configure_optimizers()
    assert restored_optimizer is not None
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    assert len(restored_optimizer.state) == len(optimizer.state)
    for old_state, new_state in zip(optimizer.state.values(), restored_optimizer.state.values()):
        assert old_state.keys() == new_state.keys()
        for key in old_state:
            if isinstance(old_state[key], torch.Tensor):
                torch.testing.assert_close(old_state[key], new_state[key])
            else:
                assert old_state[key] == new_state[key]

    # Strictly restored buffers and coefficient ordering must reproduce every
    # prediction surface once the runtime normalization is restored as well.
    restored._norm_factor = model._norm_factor
    prediction_data = data_batch.datasets[DataSet.DataSetCategory.A_SR]
    np.testing.assert_array_equal(restored.predict(prediction_data), model.predict(prediction_data))
    np.testing.assert_array_equal(
        restored.predict_secondary(prediction_data), model.predict_secondary(prediction_data)
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [pytest.param(_CONTINUATION_CONFIG, id="explicit-adaptive-neural")],
    indirect=True,
)
def test_continuation_restores_normalization_and_resumes_history(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    tmp_path,
    monkeypatch,
):
    context = function_execution_context
    data_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = _make_model(context, detector_effect, "continuation")
    optimizer = _take_one_step(model, data_batch)
    checkpoint_path = save_training_checkpoint(
        context=_checkpoint_context(tmp_path, context),
        model_name="continuation",
        model=model,
        optimizer=optimizer,
        epoch=0,
        training_history=model._training_history,
    )
    checkpoint = _torch_load(checkpoint_path)

    restored = _make_model(context, detector_effect, "continuation")
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (checkpoint_path, checkpoint),
    )
    monkeypatch.setattr(
        "neural_networks.differentiating_model.save_training_checkpoint",
        lambda **_kwargs: checkpoint_path,
    )

    history = restored.fit(data_batch)

    assert restored._epochs_executed == 2
    assert history[HistoryKeys.EPOCH.value] == [0, 1, 2]
    assert len(history[HistoryKeys.LOSS.value]) == 3
    assert restored._norm_factor._factors == model._norm_factor._factors
    assert restored._norm_factor._offsets == model._norm_factor._offsets


@pytest.mark.parametrize(
    "function_execution_context",
    [
        pytest.param(
            {
                ConfigType.DATASET: _DATASET,
                ConfigType.DETECTOR: _DETECTOR,
                ConfigType.TRAIN: Path(
                    "test/configs/train/issue018_s04_1D_cubic_binned.json"
                ),
            },
            id="cubic-bspline",
        )
    ],
    indirect=True,
)
def test_fixed_geometry_mismatch_has_contextual_error_before_state_load(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    tmp_path,
    monkeypatch,
):
    # Alter only the sidecar geometry.  The state_dict shape remains valid, so
    # this specifically guards against silently loading coefficients with the
    # wrong basis ordering/geometry.
    context = function_execution_context
    data_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = _make_model(context, detector_effect, "mismatch")
    optimizer = _take_one_step(model, data_batch)
    checkpoint_path = save_training_checkpoint(
        context=_checkpoint_context(tmp_path, context),
        model_name="mismatch",
        model=model,
        optimizer=optimizer,
        epoch=0,
        training_history=model._training_history,
    )
    metadata_path = checkpoint_metadata_path(checkpoint_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["f"]["options"]["knots"] = [-1.0, -0.25, 0.0, 0.5, 1.0]
    metadata_path.write_text(json.dumps(metadata))
    checkpoint = _torch_load(checkpoint_path)

    restored = _make_model(context, detector_effect, "mismatch")
    before = {key: value.detach().clone() for key, value in restored.state_dict().items()}
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (checkpoint_path, checkpoint),
    )
    restored_optimizer = restored.configure_optimizers()
    assert restored_optimizer is not None
    with pytest.raises(RuntimeError, match="incompatible.*f"):
        restored._load_training_checkpoint_if_requested(restored_optimizer)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, before[key])


@pytest.mark.parametrize("function_execution_context", _CASES, indirect=True)
def test_legacy_checkpoint_without_sidecar_still_loads(
    function_execution_context,
    isolated_data_generation,
    detector_effect,
    tmp_path,
    monkeypatch,
):
    context = function_execution_context
    data_batch = detector_effect.affect_batch(isolated_data_generation.get_batch())
    model = _make_model(context, detector_effect, "legacy_load")
    optimizer = _take_one_step(model, data_batch)
    checkpoint_path = save_training_checkpoint(
        context=_checkpoint_context(tmp_path, context),
        model_name="legacy_load",
        model=model,
        optimizer=optimizer,
        epoch=0,
        training_history=model._training_history,
    )
    checkpoint_metadata_path(checkpoint_path).unlink()
    checkpoint = _torch_load(checkpoint_path)

    restored = _make_model(context, detector_effect, "legacy_load")
    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (checkpoint_path, checkpoint),
    )
    restored_optimizer = restored.configure_optimizers()
    assert restored_optimizer is not None
    assert restored._load_training_checkpoint_if_requested(restored_optimizer) == 1
    assert restored._norm_factor is None
    assert restored._training_history[HistoryKeys.EPOCH.value] == [0]