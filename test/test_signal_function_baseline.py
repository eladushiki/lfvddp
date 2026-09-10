import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_tools.data_generation import DataGeneration
from data_tools.data_utils import DataSet
from frame.file_system.training_history import HistoryKeys
from neural_networks.differentiating_model import DifferentiatingModel
from test.environment import ConfigType
from train.checkpoints import _torch_load, save_training_checkpoint


_DATASET_CONFIG = Path(
    "test/configs/dataset/disjoint_1D_generated_dataset_config.json"
)
_DETECTOR_CONFIG = Path("test/configs/detector/basic_1D_detector_config.json")
_SIGNAL_PARAMETERS = (
    ("signal_region_shift_network.hidden.weight", (4, 1)),
    ("signal_region_shift_network.hidden.bias", (4,)),
    ("signal_region_shift_network.output.weight", (1, 4)),
    ("signal_region_shift_network.output.bias", (1,)),
)
_NUISANCE_PARAMETERS = {
    "binned": (("nuisance_calculation._nuisance_deltas.dimension_0", (10,)),),
    "disabled": (),
    "neural": (
        ("nuisance_calculation.network.hidden.weight", (2, 1)),
        ("nuisance_calculation.network.hidden.bias", (2,)),
        ("nuisance_calculation.network.output.weight", (1, 2)),
        ("nuisance_calculation.network.output.bias", (1,)),
    ),
}
_BASELINES = {
    "binned": {
        "initial_state": (
            "90a0115ea0c391df10acf92149ff69f64cbaf2ce635780af9d7083fee60e73e4"
        ),
        "initial_loss": "0x1.2c7c35a786bc3p+5",
        "one_step_prediction": (
            "0f9d32f0e0a6106946d012301882d777007754f1f6f1ef165befd652031744a1"
        ),
        "continued_state": (
            "3f9f232efed4524be1a2e632641db49a5492ee9dd64dd409cb1dfcdafd18c518"
        ),
        "continued_prediction": (
            "a429ff3b0c41d5ad6c5c22053902b16259ac7f10d95a1b219c916f048f4a9c0c"
        ),
    },
    "disabled": {
        "initial_state": (
            "2bf7dabca1b0d5d924df48ac90612ad14606164bc5eb7f6590e1d31ebeb1eaf5"
        ),
        "initial_loss": "0x1.2c79ca6ddaff9p+5",
        "one_step_prediction": (
            "140b6c30b651c962f9de184ea80f4bbd53efaadab9f7ff38b341836379f78984"
        ),
        "continued_state": (
            "d381ffbf3f944a4326c1dce07e66e89d655e3f3a6c8186136c8e470fcf02ff8b"
        ),
        "continued_prediction": (
            "d00587e2b81d32cf9d2539f8abc90a89bfa66b350134d45ea4b5536a66b0f954"
        ),
    },
    "neural": {
        "initial_state": (
            "1baccdf56705aa8fdc4ffeb3fe1049706d4b02aa5c419707dced28684dfe3e17"
        ),
        "initial_loss": "0x1.b4d4c43d9f3a1p+4",
        "one_step_prediction": (
            "273e1ec45147adfb2ce778ede720e9ae8cce6ff0e9195b66003f7fd407b95619"
        ),
        "continued_state": (
            "5f055727713f06153c6808270461eb9fcf4acd5320973c2be3523d3f1cd79c38"
        ),
        "continued_prediction": (
            "cad321410c527655a5945d0aa78482d0e60cfcc98a6d81b50fab9da5e646de6c"
        ),
    },
}
_CHECKPOINT_KEYS = {
    "model_name",
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "training_history",
    "array_index",
    "run_hash",
}


def _canonical_tensor_digest(labeled_tensors) -> str:
    """Hash labels, shapes, and little-endian float64 tensor values."""
    digest = hashlib.sha256()
    for name, tensor in labeled_tensors:
        values = (
            tensor
            if isinstance(tensor, np.ndarray)
            else tensor.detach().cpu().numpy()
        )
        values = np.ascontiguousarray(values, dtype="<f8")
        metadata = json.dumps(
            {"name": name, "shape": list(values.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(metadata)
        digest.update(b"\n")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _state_digest(model: DifferentiatingModel) -> str:
    return _canonical_tensor_digest(model.state_dict().items())


def _prediction_digest(model: DifferentiatingModel, data: DataSet) -> str:
    return _canonical_tensor_digest(
        (
            ("predict", model.predict(data)),
            ("predict_secondary", model.predict_secondary(data)),
            ("predict_theta", model.predict_theta(data)),
        )
    )


@pytest.fixture(autouse=True)
def preserve_deterministic_global_state():
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    number_of_threads = torch.get_num_threads()
    data_generation_instance = DataGeneration._instance
    loaded_datasets = DataGeneration._loaded_datasets

    DataGeneration._instance = None
    DataGeneration._loaded_datasets = {}
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        torch.random.set_rng_state(torch_rng_state)
        torch.use_deterministic_algorithms(
            deterministic_algorithms,
            warn_only=deterministic_warn_only,
        )
        torch.set_num_threads(number_of_threads)
        DataGeneration._instance = data_generation_instance
        DataGeneration._loaded_datasets = loaded_datasets


@pytest.mark.parametrize(
    ("function_execution_context", "nuisance_mode"),
    [
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_CONFIG,
                ConfigType.DETECTOR: _DETECTOR_CONFIG,
                ConfigType.TRAIN: Path(
                    "test/configs/train/baseline_1D_omitted_f_binned_nuisance.json"
                ),
            },
            "binned",
            id="binned-nuisance",
        ),
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_CONFIG,
                ConfigType.DETECTOR: _DETECTOR_CONFIG,
                ConfigType.TRAIN: Path(
                    "test/configs/train/baseline_1D_omitted_f_disabled_nuisance.json"
                ),
            },
            "disabled",
            id="disabled-nuisance",
        ),
        pytest.param(
            {
                ConfigType.DATASET: _DATASET_CONFIG,
                ConfigType.DETECTOR: _DETECTOR_CONFIG,
                ConfigType.TRAIN: Path(
                    "test/configs/train/baseline_1D_omitted_f_neural_nuisance.json"
                ),
            },
            "neural",
            id="neural-nuisance",
        ),
    ],
    indirect=["function_execution_context"],
)
def test_canonical_adaptive_model_contract(
    function_execution_context,
    nuisance_mode,
    isolated_data_generation,
    detector_effect,
    tmp_path,
    monkeypatch,
):
    baseline = _BASELINES[nuisance_mode]
    detected_batch = detector_effect.affect_batch(
        isolated_data_generation.get_batch()
    )
    prediction_data = detected_batch.datasets[DataSet.DataSetCategory.A_SR]

    model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="omitted_f_baseline",
        dtype=torch.float64,
        device="cpu",
    )
    expected_parameters = _NUISANCE_PARAMETERS[nuisance_mode] + _SIGNAL_PARAMETERS
    assert tuple(
        (name, tuple(parameter.shape)) for name, parameter in model.named_parameters()
    ) == expected_parameters
    assert tuple(model.state_dict()) == tuple(name for name, _ in expected_parameters)
    assert _state_digest(model) == baseline["initial_state"]

    prepared_data = model._prepare_training_data(detected_batch)
    optimizer = model.configure_optimizers()
    assert optimizer is not None
    model._set_learning_rate_for_epoch(optimizer, 0)
    loss = model(prepared_data)
    assert float(loss.detach().cpu()).hex() == baseline["initial_loss"]

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    model.nuisance_calculation.clamp_parameters()
    model._log(0, loss)

    predictions = (
        model.predict(prediction_data),
        model.predict_secondary(prediction_data),
        model.predict_theta(prediction_data),
    )
    assert tuple(prediction.shape for prediction in predictions) == (
        (9, 1),
        (9, 1),
        (9, 1),
    )
    assert _prediction_digest(model, prediction_data) == baseline["one_step_prediction"]

    denominator = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=False,
        name="omitted_f_denominator",
        dtype=torch.float64,
        device="cpu",
    )
    assert denominator.signal_region_shift_network is None
    assert tuple(
        (name, tuple(parameter.shape))
        for name, parameter in denominator.named_parameters()
    ) == _NUISANCE_PARAMETERS[nuisance_mode]
    denominator._prepare_training_data(detected_batch)
    denominator_theta = denominator.predict_theta(prediction_data)
    np.testing.assert_array_equal(
        denominator.predict(prediction_data), 1.0 + denominator_theta
    )
    np.testing.assert_array_equal(
        denominator.predict_secondary(prediction_data), 1.0 - denominator_theta
    )

    checkpoint_context = SimpleNamespace(
        training_outcomes_dir=tmp_path,
        array_index=function_execution_context.array_index,
        run_hash=function_execution_context.run_hash,
    )
    checkpoint_path = save_training_checkpoint(
        context=checkpoint_context,
        model_name="omitted_f_baseline",
        model=model,
        optimizer=optimizer,
        epoch=0,
        training_history=model._training_history,
    )
    checkpoint = _torch_load(checkpoint_path)
    assert set(checkpoint) == _CHECKPOINT_KEYS
    assert tuple(checkpoint["model_state_dict"]) == tuple(model.state_dict())
    assert checkpoint["epoch"] == 0
    assert checkpoint["optimizer_state_dict"]["state"]
    assert checkpoint["training_history"][HistoryKeys.EPOCH.value] == [0]
    checkpoint_loss = checkpoint["training_history"][HistoryKeys.LOSS.value][0]
    assert float(checkpoint_loss).hex() == baseline["initial_loss"]

    restored_model = DifferentiatingModel(
        context=function_execution_context,
        detector_effect=detector_effect,
        is_numerator=True,
        name="omitted_f_baseline",
        dtype=torch.float64,
        device="cpu",
    )
    incompatible_keys = restored_model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    assert incompatible_keys.missing_keys == []
    assert incompatible_keys.unexpected_keys == []

    monkeypatch.setattr(
        "neural_networks.differentiating_model.find_latest_training_checkpoint",
        lambda *_args, **_kwargs: (checkpoint_path, checkpoint),
    )
    monkeypatch.setattr(
        "neural_networks.differentiating_model.save_training_checkpoint",
        lambda **_kwargs: checkpoint_path,
    )
    continued_history = restored_model.fit(detected_batch)

    assert restored_model._epochs_executed == 2
    assert continued_history[HistoryKeys.EPOCH.value] == [0, 1, 2]
    assert float(continued_history[HistoryKeys.LOSS.value][0]).hex() == baseline[
        "initial_loss"
    ]
    assert _state_digest(restored_model) == baseline["continued_state"]
    assert (
        _prediction_digest(restored_model, prediction_data)
        == baseline["continued_prediction"]
    )
