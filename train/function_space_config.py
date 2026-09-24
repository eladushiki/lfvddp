"""Immutable, structural configuration for the two likelihood function spaces."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from frame.value_enum import ValueEnum


class TrainingBackend(ValueEnum):
    """The training implementation selected for the run."""

    LFVDDP = "lfvddp"
    NPLM = "nplm"


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {deepcopy(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return deepcopy(value)


@dataclass(frozen=True)
class FunctionSpaceSpec:
    """One role-neutral family name and its immutable options."""

    family: str
    options: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.family, str) or not self.family.strip():
            raise ValueError("Function-space family must be a non-empty string.")
        object.__setattr__(self, "family", self.family.strip().lower())
        if not isinstance(self.options, Mapping):
            raise ValueError(
                f"Function-space options must be a mapping, got {type(self.options).__name__}."
            )
        object.__setattr__(self, "options", _deep_freeze(self.options))

    def __repr__(self) -> str:
        return (
            "FunctionSpaceSpec("
            f"family={self.family!r}, options=<{', '.join(sorted(map(str, self.options)))}>)"
        )


def _spec_from_mapping(value: Mapping[str, Any], role: str) -> FunctionSpaceSpec:
    unknown = set(value) - {"family", "options"}
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"{role} function-space config has unknown field(s): {names}.")
    if "family" not in value:
        raise ValueError(f"{role}.family is required.")
    options = value.get("options", {})
    if options is None:
        options = {}
    if not isinstance(options, Mapping):
        raise ValueError(f"{role}.options must be a mapping, got {type(options).__name__}.")
    return FunctionSpaceSpec(family=value["family"], options=options)


def _coerce_spec(
    value: FunctionSpaceSpec | Mapping[str, Any] | None,
    role: str,
    *,
    required: bool,
) -> FunctionSpaceSpec | None:
    if value is None:
        if required:
            raise ValueError(f"{role} function-space config is required.")
        return None
    if isinstance(value, FunctionSpaceSpec):
        return FunctionSpaceSpec(value.family, value.options)
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} function-space config must be a mapping or FunctionSpaceSpec.")
    return _spec_from_mapping(value, role)


@dataclass(frozen=True)
class ResolvedFunctionSpaceConfig:
    """The selected backend plus independently configured likelihood spaces."""

    backend: TrainingBackend
    f: FunctionSpaceSpec
    nuisance: FunctionSpaceSpec | None


def resolve_dual_role_config(
    *,
    backend: TrainingBackend | str | None = None,
    f: FunctionSpaceSpec | Mapping[str, Any] | None = None,
    nuisance: FunctionSpaceSpec | Mapping[str, Any] | None = None,
) -> ResolvedFunctionSpaceConfig:
    """Resolve structural configuration without importing evaluator implementations."""

    resolved_backend = TrainingBackend.LFVDDP if backend is None else TrainingBackend.parse(backend)
    resolved_f = _coerce_spec(f, "f", required=True)
    assert resolved_f is not None
    resolved_nuisance = _coerce_spec(nuisance, "nuisance", required=False)
    if resolved_backend is TrainingBackend.NPLM and (
        resolved_f.family != "adaptive_neural"
        or (
            resolved_nuisance is not None
            and resolved_nuisance.family != "bin_indicators"
        )
    ):
        raise ValueError(
            "NPLM backend supports adaptive_neural f and optional bin_indicators nuisance; "
            "select LFVDDP for the requested function-space families."
        )
    return ResolvedFunctionSpaceConfig(
        backend=resolved_backend,
        f=resolved_f,
        nuisance=resolved_nuisance,
    )


__all__ = [
    "FunctionSpaceSpec",
    "ResolvedFunctionSpaceConfig",
    "TrainingBackend",
    "resolve_dual_role_config",
]
