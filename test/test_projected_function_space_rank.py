import numpy as np
import pytest

from neural_networks.function_spaces.rank import (
    compute_projected_function_space_rank,
    compute_rank_for_backend,
)


def test_disabled_nuisance_preserves_signal_rank():
    result = compute_projected_function_space_rank(np.eye(3), None)

    assert result.raw_f_dimension == 3
    assert result.nuisance_dimension == 0
    assert result.raw_f_rank == 3
    assert result.nuisance_rank == 0
    assert result.overlap_rank == 0
    assert result.effective_f_rank == 3


def test_disjoint_nuisance_space_does_not_reduce_signal_rank():
    f_design = np.eye(3)[:, :2]
    nuisance_design = np.array([[0.0], [0.0], [1.0]])

    result = compute_projected_function_space_rank(f_design, nuisance_design)

    assert result.raw_f_rank == 2
    assert result.nuisance_rank == 1
    assert result.overlap_rank == 0
    assert result.effective_f_rank == 2


def test_partial_overlap_has_thirteen_to_twelve_effective_rank():
    f_design = np.eye(13)
    nuisance_design = np.eye(13)[:, :1]

    result = compute_projected_function_space_rank(f_design, nuisance_design)

    assert result.raw_f_dimension == 13
    assert result.raw_f_rank == 13
    assert result.nuisance_rank == 1
    assert result.overlap_rank == 1
    assert result.effective_f_rank == 12


def test_fully_overlapping_nuisance_space_removes_signal_rank():
    f_design = np.eye(2)
    nuisance_design = f_design.copy()

    result = compute_projected_function_space_rank(f_design, nuisance_design)

    assert result.raw_f_rank == 2
    assert result.nuisance_rank == 2
    assert result.overlap_rank == 2
    assert result.effective_f_rank == 0


def test_tolerance_controls_near_singular_rank_deterministically():
    f_design = np.array([[1.0, 1.0], [0.0, 1.0e-12]])

    strict = compute_projected_function_space_rank(f_design, tolerance=1.0e-14)
    collapsed = compute_projected_function_space_rank(f_design, tolerance=1.0e-10)

    assert strict.raw_f_rank == 2
    assert collapsed.raw_f_rank == 1
    assert strict.effective_f_rank == 2
    assert collapsed.effective_f_rank == 1
    assert strict.tolerance == 1.0e-14
    assert collapsed.tolerance == 1.0e-10


def test_rank_rejects_bad_inputs_and_non_lfvddp_backend():
    with pytest.raises(ValueError, match="same row count"):
        compute_projected_function_space_rank(np.ones((3, 1)), np.ones((2, 1)))
    with pytest.raises(ValueError, match="non-finite"):
        compute_projected_function_space_rank([[np.nan]])
    with pytest.raises(ValueError, match="empirical null calibration"):
        compute_rank_for_backend(np.eye(2), backend="nplm")
