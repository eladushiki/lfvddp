import pytest

from train.function_space_config import (
    FunctionSpaceFamily,
    FunctionSpaceSpec,
    ResolvedFunctionSpaceConfig,
    RoleState,
    TrainingBackend,
    resolve_dual_role_config,
)
from train.train_config import TrainConfig


def _binned_role():
    return {
        "family": "bin_indicators",
        "options": {"minima": [0.0], "maxima": [10.0], "number_of_bins": [4]},
    }


def test_canonical_roles_use_independent_immutable_options():
    f_options = {"hidden_layer_nodes": 4, "geometry": {"width": 1.0}}
    nuisance_options = {"minima": [0.0], "maxima": [1.0], "number_of_bins": [2]}
    resolved = resolve_dual_role_config(
        backend="lfvddp",
        f={"family": "adaptive_neural", "options": f_options},
        nuisance={"family": "bin_indicators", "options": nuisance_options},
    )

    assert isinstance(resolved, ResolvedFunctionSpaceConfig)
    assert resolved.backend is TrainingBackend.LFVDDP
    assert resolved.f.family is FunctionSpaceFamily.ADAPTIVE_NEURAL
    assert resolved.nuisance.family is FunctionSpaceFamily.BIN_INDICATORS
    assert resolved.f.options is not resolved.nuisance.options
    f_options["geometry"]["width"] = 9.0
    nuisance_options["minima"].append(-1.0)
    assert resolved.f.options["geometry"]["width"] == 1.0
    assert resolved.nuisance.options["minima"] == (0.0,)
    with pytest.raises(TypeError):
        resolved.f.options["new"] = "not allowed"


def test_disabled_is_nuisance_only_state():
    resolved = resolve_dual_role_config(
        f={"family": "adaptive_neural", "options": {}},
        nuisance={"state": "disabled"},
    )
    assert resolved.nuisance.state is RoleState.DISABLED
    assert resolved.nuisance.family is None
    with pytest.raises(ValueError, match="f role is disabled"):
        resolve_dual_role_config(f={"state": "disabled"}, nuisance=_binned_role())


def test_legacy_omitted_fields_resolve_current_defaults():
    config = TrainConfig(
        train__epochs=100,
        train__number_of_epochs_for_checkpoint=10,
        train__nn_inner_layer_nodes=4,
        train__nuisance_binning_minima=0,
        train__nuisance_binning_maxima=10,
        train__nuisance_binning_number_of_bins=10,
    )
    resolved = config.train__function_space_config
    assert resolved.compatibility_source == "legacy"
    assert resolved.f.family is FunctionSpaceFamily.ADAPTIVE_NEURAL
    assert resolved.nuisance.family is FunctionSpaceFamily.BIN_INDICATORS
    assert resolved.nuisance.state is RoleState.ENABLED

    disabled = TrainConfig(
        train__epochs=100,
        train__number_of_epochs_for_checkpoint=10,
        train__nn_inner_layer_nodes=4,
        train__data_is_train_for_nuisances=False,
        train__nuisance_binning_minima=0,
        train__nuisance_binning_maxima=10,
        train__nuisance_binning_number_of_bins=10,
    )
    assert disabled.train__function_space_config.nuisance.state is RoleState.DISABLED

    neural = TrainConfig(
        train__epochs=100,
        train__number_of_epochs_for_checkpoint=10,
        train__nn_inner_layer_nodes=4,
        train__nuisance_is_neural_network=True,
        train__nuisance_nn_inner_layer_nodes=2,
    )
    assert neural.train__function_space_config.nuisance.family is FunctionSpaceFamily.ADAPTIVE_NEURAL
    assert neural.train__function_space_config.nuisance.options["hidden_layer_nodes"] == 2

    canonical = TrainConfig(
        train__epochs=100,
        train__number_of_epochs_for_checkpoint=10,
        train__nn_inner_layer_nodes=4,
        train__function_space={
            "backend": "lfvddp",
            "f": {"family": "adaptive_neural", "options": {"hidden_layer_nodes": 8}},
            "nuisance": _binned_role(),
        },
    )
    assert canonical.resolved_function_space_config.compatibility_source == "canonical"
    assert canonical.train__f_function_space_spec.options["hidden_layer_nodes"] == 8

    split_roles = TrainConfig(
        train__epochs=100,
        train__number_of_epochs_for_checkpoint=10,
        train__nn_inner_layer_nodes=4,
        train__backend="lfvddp",
        train__f_function_space={"family": "adaptive_neural", "options": {}},
        train__nuisance_function_space=_binned_role(),
    )
    assert split_roles.train__function_space_config.compatibility_source == "canonical"


def test_backend_axis_remains_orthogonal_to_function_families():
    resolved = resolve_dual_role_config(
        backend="nplm",
        legacy_f_options={"hidden_layer_nodes": 4},
        legacy_nuisance_options={"minima": [0], "maxima": [1], "number_of_bins": [2]},
    )
    assert resolved.backend is TrainingBackend.NPLM
    assert resolved.f.family is FunctionSpaceFamily.ADAPTIVE_NEURAL
    with pytest.raises(ValueError, match="conflicts with legacy train__like_NPLM"):
        TrainConfig(
            train__epochs=100,
            train__number_of_epochs_for_checkpoint=10,
            train__nn_inner_layer_nodes=4,
            train__like_NPLM=True,
            train__backend="lfvddp",
            train__nuisance_binning_minima=0,
            train__nuisance_binning_maxima=1,
            train__nuisance_binning_number_of_bins=2,
        )


def test_invalid_configs_are_contextual_and_future_families_are_declared():
    assert FunctionSpaceFamily.CUBIC_BSPLINE.value == "cubic_bspline"
    with pytest.raises(ValueError, match="nuisance.options must be a mapping"):
        resolve_dual_role_config(
            f={"family": "adaptive_neural", "options": {}},
            nuisance={"family": "bin_indicators", "options": []},
        )
    with pytest.raises(ValueError, match="nuisance family 'bin_indicators' requires"):
        resolve_dual_role_config(
            f={"family": "adaptive_neural", "options": {}},
            nuisance={"family": "bin_indicators", "options": {}},
        )
    with pytest.raises(ValueError, match="basis must be 'legendre' or 'chebyshev'"):
        resolve_dual_role_config(
            f={"family": "orthogonal_polynomial", "options": {
                "basis": "fourier", "maximum_degree": 2, "domain": [0, 1]
            }},
            nuisance=_binned_role(),
        )


def test_repr_is_sanitized():
    resolved = resolve_dual_role_config(
        f={"family": "adaptive_neural", "options": {"api_token": "secret-value"}},
        nuisance=_binned_role(),
    )
    assert "secret-value" not in repr(resolved)
    assert "redacted" in repr(resolved)
    assert "adaptive_neural" in repr(resolved)
