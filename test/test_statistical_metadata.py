import json

import numpy as np

from neural_networks.function_spaces.projected_rank import compute_projected_function_space_rank
from train.function_space_config import resolve_dual_role_config
from train.statistical_metadata import (
    EMPIRICAL_NULL_CALIBRATION,
    WILKS_CALIBRATION,
    build_statistical_metadata,
)


def _deterministic_f():
    return {
        "family": "orthogonal_polynomial",
        "options": {
            "basis": "legendre",
            "maximum_degree": 2,
            "domain": [[0.0, 1.0]],
        },
    }


def _binned_nuisance():
    return {
        "family": "bin_indicators",
        "options": {"minima": [0.0], "maxima": [1.0], "number_of_bins": [2]},
    }


def test_deterministic_metadata_is_serializable_and_uses_effective_rank_for_wilks():
    config = resolve_dual_role_config(
        backend="lfvddp",
        f=_deterministic_f(),
        nuisance=_binned_nuisance(),
    )
    rank = compute_projected_function_space_rank(
        np.eye(4)[:, :3], np.eye(4)[:, 3:]
    )

    metadata = build_statistical_metadata(config, rank)

    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["mode"] == metadata["backend"] == "lfvddp"
    assert metadata["f_family"] == "orthogonal_polynomial"
    assert metadata["f_state"] == "enabled"
    assert metadata["nuisance_family"] == "bin_indicators"
    assert metadata["nuisance_state"] == "enabled"
    assert metadata["f_feature_count"] == 3
    assert metadata["nuisance_feature_count"] == 1
    assert metadata["effective_f_rank"] == rank.effective_f_rank
    assert metadata["tolerance"] == rank.tolerance
    assert metadata["f_regularity"] == "deterministic"
    assert metadata["calibration_policy"] == WILKS_CALIBRATION


def test_adaptive_f_metadata_requires_empirical_null_calibration():
    config = resolve_dual_role_config(
        backend="lfvddp",
        f={"family": "adaptive_neural", "options": {"hidden_layer_nodes": 4}},
        nuisance=_binned_nuisance(),
    )
    rank = compute_projected_function_space_rank(np.eye(3), np.eye(3)[:, :1])

    metadata = build_statistical_metadata(config, rank)

    assert metadata["f_family"] == "adaptive_neural"
    assert metadata["f_regularity"] == "adaptive"
    assert metadata["regularity"] == "nonregular"
    assert metadata["calibration_policy"] == EMPIRICAL_NULL_CALIBRATION


def test_disabled_nuisance_is_explicit_in_metadata():
    config = resolve_dual_role_config(
        backend="lfvddp",
        f=_deterministic_f(),
        nuisance={"state": "disabled"},
    )
    rank = compute_projected_function_space_rank(np.eye(3), None)

    metadata = build_statistical_metadata(config, rank)

    assert metadata["nuisance_family"] is None
    assert metadata["nuisance_state"] == "disabled"
    assert metadata["nuisance_feature_count"] == 0
    assert metadata["nuisance_regularity"] is None
    assert metadata["calibration_policy"] == WILKS_CALIBRATION


def test_nplm_backend_is_empirical_null():
    config = resolve_dual_role_config(
        backend="nplm",
        f={"family": "adaptive_neural", "options": {"hidden_layer_nodes": 4}},
        nuisance={"state": "disabled"},
    )
    rank = compute_projected_function_space_rank(np.eye(3), None)

    metadata = build_statistical_metadata(config, rank)

    assert metadata["mode"] == metadata["backend"] == "nplm"
    assert metadata["f_family"] == "adaptive_neural"
    assert metadata["calibration_policy"] == EMPIRICAL_NULL_CALIBRATION
