"""Serializable statistical provenance for a resolved function-space model.

This module intentionally only describes the fitted statistical model.  It does
not persist checkpoints or compute design matrices; callers provide the resolved
role configuration and the rank diagnostic computed from those matrices.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

from frame.value_enum import ValueEnum
from neural_networks.function_spaces.projected_rank import (
    ProjectedFunctionSpaceRank,
    compute_rank_for_backend,
)
from train.function_space_config import (
    FunctionSpaceFamily,
    FunctionSpaceSpec,
    ResolvedFunctionSpaceConfig,
    TrainingBackend,
)

if TYPE_CHECKING:
    from data_tools.data_generation import DataBatch
    from neural_networks.differentiating_model import DifferentiatingModel


class CalibrationPolicy(ValueEnum):
    """Null-calibration choices for the fitted function space."""

    WILKS = "wilks"
    EMPIRICAL_NULL = "empirical-null"

# These values are the role-neutral FunctionSpaceMetadata regularity labels.  A
# family is mapped here rather than by importing an evaluator, keeping metadata
# available at configuration and aggregation boundaries.
_FAMILY_REGULARITY = {
    FunctionSpaceFamily.ADAPTIVE_NEURAL: "adaptive",
    FunctionSpaceFamily.BIN_INDICATORS: "piecewise_constant",
    FunctionSpaceFamily.CUBIC_BSPLINE: "deterministic",
    FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL: "deterministic",
    FunctionSpaceFamily.FIXED_SIGMOID: "deterministic",
    FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS: "deterministic",
}


def _value(value: Any) -> Any:
    """Return the JSON-facing value of a value enum or ordinary object."""
    return getattr(value, "value", value)


def _regularity(spec: FunctionSpaceSpec) -> str | None:
    """Return the structural regularity of one enabled role."""
    if not spec.enabled:
        return None
    # Resolved specs are validated before reaching this boundary.  The fallback
    # keeps this diagnostic useful for forward-compatible family enum members.
    return _FAMILY_REGULARITY.get(spec.family, "unknown")


def _calibration_policy(config: ResolvedFunctionSpaceConfig) -> CalibrationPolicy:
    """Choose the calibration policy from the selected model families."""
    f_regularity = _regularity(config.f)
    nuisance_regularity = _regularity(config.nuisance)
    has_adaptive_role = "adaptive" in {f_regularity, nuisance_regularity}
    if config.backend is TrainingBackend.NPLM or has_adaptive_role:
        return CalibrationPolicy.EMPIRICAL_NULL
    return CalibrationPolicy.WILKS


def build_statistical_metadata(
    resolved_config: ResolvedFunctionSpaceConfig,
    projected_rank: Optional[ProjectedFunctionSpaceRank],
) -> dict[str, Any]:
    """Build JSON-serializable statistical metadata for a resolved model.

    Feature counts are taken from the actual design matrices represented by the
    rank diagnostic, not inferred from generic neural parameter counts.  The
    returned dictionary contains only JSON primitives, lists, and nested
    dictionaries, so it can be passed directly to :func:`json.dumps`.
    """
    if not isinstance(resolved_config, ResolvedFunctionSpaceConfig):
        raise TypeError("resolved_config must be a ResolvedFunctionSpaceConfig.")

    f_regularity = _regularity(resolved_config.f)
    nuisance_regularity = _regularity(resolved_config.nuisance)
    calibration_policy = _calibration_policy(resolved_config)
    is_regular = calibration_policy is CalibrationPolicy.WILKS

    rank_metadata = (
        _json_value(asdict(projected_rank)) if projected_rank is not None else {}
    )

    def rank_value(name: str) -> Any:
        return rank_metadata.get(name)

    # Keep role and rank provenance flat so this can be consumed by existing
    # aggregation/reporting code without knowing evaluator implementation types.
    return {
        "mode": _value(resolved_config.backend),
        "backend": _value(resolved_config.backend),
        "f_family": _value(resolved_config.f.family),
        "f_state": _value(resolved_config.f.state),
        "nuisance_family": _value(resolved_config.nuisance.family),
        "nuisance_state": _value(resolved_config.nuisance.state),
        "f_feature_count": rank_value("raw_f_dimension"),
        "nuisance_feature_count": rank_value("nuisance_dimension"),
        "raw_f_dimension": rank_value("raw_f_dimension"),
        "nuisance_dimension": rank_value("nuisance_dimension"),
        "raw_f_rank": rank_value("raw_f_rank"),
        "nuisance_rank": rank_value("nuisance_rank"),
        "overlap_rank": rank_value("overlap_rank"),
        "effective_f_rank": rank_value("effective_f_rank"),
        # The numerator also fits the nuisance parameters.  The test statistic
        # compares it with a denominator that fits the same nuisance model, so
        # only the projected f rank belongs to its chi-square reference.
        "statistic_degrees_of_freedom": rank_value("effective_f_rank"),
        "degrees_of_freedom": rank_value("degrees_of_freedom"),
        "tolerance": rank_value("tolerance"),
        "f_regularity": f_regularity,
        "nuisance_regularity": nuisance_regularity,
        "regularity": "regular" if is_regular else "nonregular",
        "calibration_policy": calibration_policy.value,
    }


def build_model_statistical_metadata(
    model: "DifferentiatingModel",
    data: "DataBatch",
) -> dict[str, Any]:
    """Build persisted calibration metadata from one trained numerator model."""

    resolved_config = model._function_space_config
    if _calibration_policy(resolved_config) is CalibrationPolicy.EMPIRICAL_NULL:
        return build_statistical_metadata(resolved_config, None)

    f_design, nuisance_design = model.statistical_design_matrices(data)
    if f_design is None:
        raise RuntimeError(
            "A regular function-space model did not provide a statistical design."
        )
    projected_rank = compute_rank_for_backend(
        f_design,
        nuisance_design,
        backend=resolved_config.backend,
    )
    return build_statistical_metadata(resolved_config, projected_rank)


def _json_value(value: Any) -> Any:
    """Normalize numpy scalars/tuples and dataclass values to JSON primitives."""
    if hasattr(value, "item") and callable(value.item):
        return _json_value(value.item())
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if is_dataclass(value):
        return _json_value(asdict(value))
    return value


__all__ = [
    "CalibrationPolicy",
    "build_model_statistical_metadata",
    "build_statistical_metadata",
]
