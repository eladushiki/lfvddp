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


def test_train_config_requires_canonical_role_mappings():
    assert "train__nuisance" in TrainConfig.__dataclass_fields__
    with pytest.raises(ValueError, match="train__nuisance must define"):
        TrainConfig(
            train__epochs=100,
            train__number_of_epochs_for_checkpoint=10,
            train__nn_inner_layer_nodes=4,
            train__f={"family": "adaptive_neural", "options": {}},
        )

def test_backend_axis_remains_orthogonal_to_function_families():
    resolved = resolve_dual_role_config(
        backend="nplm",
        f={"family": "adaptive_neural", "options": {"input_dimension": 1, "hidden_layer_nodes": 4}},
        nuisance=_binned_role(),
    )
    assert resolved.backend is TrainingBackend.NPLM
    assert resolved.f.family is FunctionSpaceFamily.ADAPTIVE_NEURAL
    with pytest.raises(ValueError, match="supports adaptive f and binned nuisance"):
        resolve_dual_role_config(
            backend="nplm",
            f={"family": "cubic_bspline", "options": {"knots": [0, 0, 0, 0, 1, 1, 1, 1]}},
            nuisance=_binned_role(),
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
