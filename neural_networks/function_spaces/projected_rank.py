"""Identifiable rank of one function space after nuisance projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import torch

from train.function_space_config import TrainingBackend


@dataclass(frozen=True)
class ProjectedFunctionSpaceRank:
    """Rank diagnostics for a signal design space projected off nuisance."""

    raw_f_dimension: int
    nuisance_dimension: int
    raw_f_rank: int
    nuisance_rank: int
    overlap_rank: int
    effective_f_rank: int
    projected_singular_values: tuple[float, ...]
    tolerance: float


def _matrix(value: Any, name: str) -> npt.NDArray[np.float64]:
    if value is None:
        return np.empty((0, 0), dtype=np.float64)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.ndim != 2:
        raise ValueError(f"{name} design matrix must be two-dimensional.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} design matrix contains non-finite values.")
    return np.ascontiguousarray(result)


def _svd_rank(matrix: npt.NDArray[np.float64], tolerance: Optional[float]) -> tuple[int, float, npt.NDArray[np.float64]]:
    if matrix.size == 0 or matrix.shape[1] == 0:
        return 0, float(tolerance or 0.0), np.empty(0, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    default_tolerance = max(matrix.shape) * np.finfo(np.float64).eps * scale
    threshold = default_tolerance if tolerance is None else float(tolerance)
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("rank tolerance must be finite and non-negative.")
    return int(np.count_nonzero(singular_values > threshold)), threshold, singular_values


def compute_projected_function_space_rank(
    f_design: Any,
    nuisance_design: Any = None,
    *,
    tolerance: Optional[float] = None,
) -> ProjectedFunctionSpaceRank:
    """Compute identifiable signal rank after orthogonal nuisance projection.

    Matrices are interpreted as rows of observations and columns of basis
    functions.  A disabled nuisance role is represented by ``None`` or an
    empty matrix.  The projection is built from the numerically stable SVD
    nuisance span; coefficient counts are never used as a rank substitute.
    """
    f_matrix = _matrix(f_design, "f")
    nuisance_matrix = _matrix(nuisance_design, "nuisance")
    if nuisance_matrix.size and f_matrix.size and nuisance_matrix.shape[0] != f_matrix.shape[0]:
        raise ValueError("f and nuisance design matrices must have the same row count.")

    raw_rank, threshold, _ = _svd_rank(f_matrix, tolerance)
    nuisance_rank, nuisance_threshold, _ = _svd_rank(nuisance_matrix, tolerance)
    if tolerance is None:
        threshold = max(threshold, nuisance_threshold)

    if nuisance_rank:
        u, _, _ = np.linalg.svd(nuisance_matrix, full_matrices=False)
        nuisance_basis = u[:, :nuisance_rank]
        residual = f_matrix - nuisance_basis @ (nuisance_basis.T @ f_matrix)
    else:
        residual = f_matrix

    effective_rank, _, projected_singular_values = _svd_rank(residual, threshold)
    if nuisance_matrix.size:
        combined = np.concatenate((f_matrix, nuisance_matrix), axis=1)
        combined_rank, _, _ = _svd_rank(combined, threshold)
    else:
        combined_rank = raw_rank
    overlap_rank = max(0, raw_rank + nuisance_rank - combined_rank)

    result = ProjectedFunctionSpaceRank(
        raw_f_dimension=f_matrix.shape[1],
        nuisance_dimension=nuisance_matrix.shape[1],
        raw_f_rank=raw_rank,
        nuisance_rank=nuisance_rank,
        overlap_rank=overlap_rank,
        effective_f_rank=effective_rank,
        projected_singular_values=tuple(float(value) for value in projected_singular_values),
        tolerance=float(threshold),
    )
    return result


# Concise aliases for callers using the scientific term rather than the full
# function name.  They intentionally resolve to one production implementation.
projected_function_space_rank = compute_projected_function_space_rank
compute_projected_rank = compute_projected_function_space_rank


def compute_rank_for_backend(
    f_design: Any,
    nuisance_design: Any = None,
    *,
    backend: TrainingBackend | str = TrainingBackend.LFVDDP,
    tolerance: Optional[float] = None,
) -> ProjectedFunctionSpaceRank:
    """Compute rank only for deterministic LFVDDP design spaces.

    Adaptive neural spaces and NPLM do not have a fixed design-space rank and
    therefore must be calibrated empirically rather than assigned Wilks
    degrees of freedom by this helper.
    """
    resolved = TrainingBackend.from_value(backend)
    if resolved is not TrainingBackend.LFVDDP:
        raise ValueError(
            f"Projected design-space rank is unavailable for backend {resolved.value!r}; "
            """use empirical null calibration instead."""
        )
    return compute_projected_function_space_rank(
        f_design, nuisance_design, tolerance=tolerance
    )
