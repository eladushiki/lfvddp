import pytest

from train.function_space_config import (
    FunctionSpaceSpec,
    TrainingBackend,
    resolve_dual_role_config,
)
from train.train_config import TrainConfig


def _binned_spec():
    return {
        "family": "bin_indicators",
        "options": {"minima": [0.0], "maxima": [10.0], "number_of_bins": [4]},
    }


def _adaptive_spec():
    return {
        "family": "adaptive_neural",
        "options": {"input_dimension": 1, "hidden_layer_nodes": 4},
    }


def test_canonical_roles_use_independent_immutable_options():
    options = {"input_dimension": 1, "hidden_layer_nodes": 4, "geometry": {"width": 1.0}}
    resolved = resolve_dual_role_config(f={"family": "adaptive_neural", "options": options}, nuisance=_binned_spec())

    assert resolved.backend is TrainingBackend.LFVDDP
    assert resolved.f.family == "adaptive_neural"
    assert resolved.nuisance is not None
    assert resolved.nuisance.family == "bin_indicators"
    options["geometry"]["width"] = 9.0
    assert resolved.f.options["geometry"]["width"] == 1.0
    with pytest.raises(TypeError):
        resolved.f.options["new"] = "not allowed"


def test_null_nuisance_is_the_only_disabled_canonical_form():
    resolved = resolve_dual_role_config(f=_adaptive_spec(), nuisance=None)
    assert resolved.nuisance is None
    with pytest.raises(ValueError, match="f function-space config is required"):
        resolve_dual_role_config(f=None, nuisance=None)
    with pytest.raises(ValueError, match="unknown field"):
        resolve_dual_role_config(f=_adaptive_spec(), nuisance={"state": "disabled"})


def test_backend_is_the_only_runtime_selector_and_validates_supported_spaces():
    resolved = resolve_dual_role_config(backend="nplm", f=_adaptive_spec(), nuisance=_binned_spec())
    assert resolved.backend is TrainingBackend.NPLM
    with pytest.raises(ValueError, match="NPLM backend supports"):
        resolve_dual_role_config(
            backend="nplm",
            f={"family": "cubic_bspline", "options": {"knots": [0, 1, 2]}},
            nuisance=None,
        )


def test_family_options_are_validated_by_train_configuration():
    with pytest.raises(ValueError, match="positive input_dimension"):
        TrainConfig(
            train__epochs=100,
            train__number_of_epochs_for_checkpoint=10,
            train__f={"family": "adaptive_neural", "options": {"hidden_layer_nodes": 4}},
            train__nuisance=None,
        )


def test_spec_repr_redacts_option_values():
    spec = FunctionSpaceSpec("adaptive_neural", {"api_token": "secret-value"})
    assert "secret-value" not in repr(spec)
    assert "api_token" in repr(spec)
