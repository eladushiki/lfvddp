"""Serialization and compatibility checks for function-space checkpoint sidecars."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

from data_tools.data_utils import ShiftAndNormalizationFactor
from neural_networks.function_spaces.base import SCALAR_OUTPUT_DIMENSION
from train.function_space_config import ResolvedFunctionSpaceConfig


def checkpoint_value(value: Any) -> Any:
    """Convert configuration and normalization values to JSON primitives."""

    if isinstance(value, Mapping):
        return {str(key): checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [checkpoint_value(item) for item in value]
    if isinstance(value, Enum):
        return checkpoint_value(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported checkpoint metadata value {type(value).__name__}.")


def _compatibility_fingerprint(compatibility: Mapping[str, Any]) -> str:
    """Hash the canonical structural portion of checkpoint metadata."""

    return hashlib.sha256(
        json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_checkpoint_metadata(
    *,
    model_name: str,
    is_numerator: bool,
    resolved_config: ResolvedFunctionSpaceConfig,
    normalization_factor: ShiftAndNormalizationFactor | None,
) -> dict[str, Any]:
    """Build the sidecar payload without inspecting model internals."""

    compatibility = {
        "model_name": model_name,
        "is_numerator": is_numerator,
        "backend": checkpoint_value(resolved_config.backend),
        "f": {
            "family": checkpoint_value(resolved_config.f.family),
            "options": checkpoint_value(resolved_config.f.options),
        },
        "nuisance": (
            None
            if resolved_config.nuisance is None
            else {
                "family": checkpoint_value(resolved_config.nuisance.family),
                "options": checkpoint_value(resolved_config.nuisance.options),
            }
        ),
    }
    for role in ("f", "nuisance"):
        spec = compatibility[role]
        if (
            spec is not None
            and spec["family"] == "adaptive_neural"
            and normalization_factor is not None
        ):
            # Preserve the historical fingerprint's structural input width,
            # now derived from the pooled observable map rather than options.
            spec["options"]["input_dimension"] = normalization_factor.n_dim
    fingerprint = _compatibility_fingerprint(compatibility)
    return {
        "format_version": 3,
        **compatibility,
        "config_fingerprint": fingerprint,
        "normalization_factor": (
            None
            if normalization_factor is None
            else checkpoint_value(normalization_factor.to_mapping())
        ),
    }


def validate_checkpoint_metadata(
    *,
    checkpoint_path: Path,
    model_name: str,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    """Reject a structurally incompatible sidecar before loading model state."""

    expected_compatibility = {
        key: value for key, value in expected.items() if key != "normalization_factor"
    }
    actual_compatibility = {key: actual.get(key) for key in expected_compatibility}
    # Old checkpoints could explicitly record the already-scalar output.
    # Canonicalize that redundant hint without relaxing input-width checks.
    actual_compatibility = checkpoint_value(actual_compatibility)
    removed_output_hint = False
    for role in ("f", "nuisance"):
        spec = actual_compatibility.get(role)
        if isinstance(spec, dict) and spec.get("family") == "adaptive_neural":
            options = spec.get("options", {})
            if (
                isinstance(options, dict)
                and options.get("output_dimension") == SCALAR_OUTPUT_DIMENSION
            ):
                options.pop("output_dimension")
                removed_output_hint = True
    if removed_output_hint:
        actual_compatibility["config_fingerprint"] = _compatibility_fingerprint(
            {
                key: actual_compatibility[key]
                for key in ("model_name", "is_numerator", "backend", "f", "nuisance")
            }
        )
    if actual_compatibility == expected_compatibility:
        return
    differences = [
        key
        for key in sorted(set(expected_compatibility) | set(actual_compatibility))
        if expected_compatibility.get(key) != actual_compatibility.get(key)
    ]
    changed = ", ".join(differences) or "unknown metadata"
    raise RuntimeError(
        f"Checkpoint {checkpoint_path} is incompatible with {model_name}: "
        f"{changed} differs. Refusing to load before state_dict validation."
    )


def normalization_from_checkpoint_metadata(
    checkpoint_path: Path,
    metadata: Mapping[str, Any],
) -> ShiftAndNormalizationFactor | None:
    """Restore validated normalization data from an optional checkpoint sidecar."""

    normalization = metadata.get("normalization_factor")
    if normalization is None:
        return None
    if not isinstance(normalization, Mapping):
        raise RuntimeError(
            f"Checkpoint metadata {checkpoint_path} has an invalid normalization_factor."
        )
    try:
        return ShiftAndNormalizationFactor.from_mapping(normalization)
    except (TypeError, ValueError, AssertionError) as error:
        raise RuntimeError(
            f"Checkpoint metadata {checkpoint_path} has invalid normalization values."
        ) from error
