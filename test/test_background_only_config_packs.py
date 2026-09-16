"""Executable background-only cluster packs for every supported f family."""

from pathlib import Path

import pytest

from frame.command_line.handle_args import create_config_from_paths
from train.function_space_config import FunctionSpaceFamily


_PACK_ROOT = Path(__file__).parents[1] / "configs" / "background-only"
_COMMON_PATHS = tuple(sorted((_PACK_ROOT / "common").glob("*.json")))
_EXPECTED_OPTIONS = {
    "adaptive_neural": {"input_dimension": 2, "hidden_layer_nodes": 4},
    "bin_indicators": {
        "minima": (0.0, 0.0),
        "maxima": (10.0, 10.0),
        "number_of_bins": (2, 2),
    },
    "cubic_bspline": {
        "knots": ((0.0, 2.5, 5.0, 7.5, 10.0),) * 2,
    },
    "fixed_sigmoid": {
        "centers": ((2.5, 2.5), (7.5, 7.5)),
        "widths": (1.75, 1.75),
    },
    "orthogonal_polynomial": {
        "basis": "legendre",
        "maximum_degree": 3,
        "domain": (0.0, 10.0),
    },
}


@pytest.mark.parametrize("mode", tuple(_EXPECTED_OPTIONS))
def test_background_only_cluster_pack_composes_with_physical_geometry(mode):
    """Keep the executable comparison packs in physical observable units."""

    config = create_config_from_paths(
        [*_COMMON_PATHS, _PACK_ROOT / mode / "train_config.json"]
    )
    resolved = config.train__resolved_function_space_config

    assert resolved.f.family is FunctionSpaceFamily(mode)
    assert resolved.f.options == _EXPECTED_OPTIONS[mode]
    assert resolved.nuisance.family is FunctionSpaceFamily.BIN_INDICATORS
    assert resolved.nuisance.options == _EXPECTED_OPTIONS["bin_indicators"]
