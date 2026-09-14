"""Serialization and compatibility checks for function-space checkpoint sidecars."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from data_tools.data_utils import ShiftAndNormalizationFactor
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


def build_checkpoint_metadata(
    *,
    model_name: str,
    is_numerator: bool,
    resolved_config: ResolvedFunctionSpaceConfig,
    normalization_factor: Optional[ShiftAndNormalizationFactor],
) -> dict[str, Any]:
    """Build the sidecar payload without inspecting model internals."""

    compatibility = {
        "model_name": model_name,
        "is_numerator": is_numerator,
        "backend": checkpoint_value(resolved_config.backend),
        "f": {
            "family": checkpoint_value(resolved_config.f.family),
            "state": checkpoint_value(resolved_config.f.state),
            "options": checkpoint_value(resolved_config.f.options),
        },
        "nuisance": {
            "family": checkpoint_value(resolved_config.nuisance.family),
            "state": checkpoint_value(resolved_config.nuisance.state),
            "options": checkpoint_value(resolved_config.nuisance.options),
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(compatibility, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "format_version": 1,
        **compatibility,
        "config_fingerprint": fingerprint,
        "normalization_factor": (
            None
            if normalization_factor is None
            else {
                "factors": checkpoint_value(normalization_factor._factors),
                "offsets": checkpoint_value(normalization_factor._offsets),
            }
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
    actual_compatibility = {
        key: actual.get(key) for key in expected_compatibility
    }
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
) -> Optional[ShiftAndNormalizationFactor]:
    """Restore validated normalization data from an optional checkpoint sidecar."""

    normalization = metadata.get("normalization_factor")
    if normalization is None:
        return None
    if not isinstance(normalization, Mapping) or not {"factors", "offsets"} <= set(normalization):
        raise RuntimeError(
            f"Checkpoint metadata {checkpoint_path} has an invalid normalization_factor."
        )
    factors = normalization["factors"]
    offsets = normalization["offsets"]
    if not isinstance(factors, Mapping) or not isinstance(offsets, Mapping):
        raise RuntimeError(
            f"Checkpoint metadata {checkpoint_path} has invalid normalization mappings."
        )
    try:
        return ShiftAndNormalizationFactor(
            {str(key): float(value) for key, value in factors.items()},
            {str(key): float(value) for key, value in offsets.items()},
        )
    except (TypeError, ValueError, AssertionError) as error:
        raise RuntimeError(
            f"Checkpoint metadata {checkpoint_path} has invalid normalization values."
        ) from error
