"""Canonical configuration for the two learned likelihood roles.

This module deliberately contains no model or evaluator code.  It translates the
flat configuration used by older config packs into one immutable description
that both role adapters can consume.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

from frame.value_enum import ValueEnum

class TrainingBackend(ValueEnum):
    """Training implementation, orthogonal to mathematical function family."""

    LFVDDP = "lfvddp"
    NPLM = "nplm"

    @classmethod
    def from_value(cls, value: TrainingBackend | str | None) -> TrainingBackend:
        if value is None:
            return cls.LFVDDP
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError(
                f"Unknown training backend {value!r}; expected one of: lfvddp, nplm."
            ) from error


class FunctionSpaceFamily(ValueEnum):
    """Closed set of reusable function-space families.

    The later families are represented here so they can be declared and
    validated now.  Their evaluators are intentionally implemented by the
    subsequent shared-family slice, not by this configuration module.
    """

    ADAPTIVE_NEURAL = "adaptive_neural"
    BIN_INDICATORS = "bin_indicators"
    CUBIC_BSPLINE = "cubic_bspline"
    ORTHOGONAL_POLYNOMIAL = "orthogonal_polynomial"
    FIXED_SIGMOID = "fixed_sigmoid"
    GAUSSIAN_RADIAL_BASIS = "gaussian_radial_basis"

    @classmethod
    def from_value(cls, value: FunctionSpaceFamily | str) -> FunctionSpaceFamily:
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown function-space family {value!r}; expected one of: {choices}."
            ) from error


class FunctionSpaceRole(ValueEnum):
    """The two typed likelihood roles that share a function-space configuration shape."""

    F = "f"
    NUISANCE = "nuisance"

    @classmethod
    def from_value(cls, value: FunctionSpaceRole | str) -> FunctionSpaceRole:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown function-space role {value!r}; expected one of: {choices}."
            ) from error


class RoleState(ValueEnum):
    """Whether a role participates in the model.

    ``DISABLED`` is valid only for nuisance.  In particular, f is not disabled
    to express the denominator's omission; that remains a model-role semantic.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"

    @classmethod
    def from_value(cls, value: RoleState | str | bool | None) -> RoleState:
        if value is None:
            return cls.ENABLED
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError(
                f"Unknown role state {value!r}; expected 'enabled' or 'disabled'."
            ) from error


def _deep_freeze(value: Any) -> Any:
    """Copy a config value and recursively make containers immutable."""
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


def _role_spec_from_mapping(mapping: Mapping[str, Any], role: str) -> "FunctionSpaceSpec":
    unknown = set(mapping) - {"family", "options", "state"}
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"{role} function-space config has unknown field(s): {names}.")
    state = RoleState.from_value(mapping.get("state"))
    raw_family = mapping.get("family")
    raw_options = mapping.get("options", {})
    if raw_options is None:
        raw_options = {}
    if not isinstance(raw_options, Mapping):
        raise ValueError(f"{role}.options must be a mapping, got {type(raw_options).__name__}.")
    if state is RoleState.DISABLED:
        if raw_family is not None or raw_options:
            raise ValueError(
                f"{role} role is disabled; disabled roles must not define family or options."
            )
        family = None
    else:
        if raw_family is None:
            raise ValueError(f"{role}.family is required when the role is enabled.")
        family = FunctionSpaceFamily.from_value(raw_family)
    spec = FunctionSpaceSpec(family=family, options=raw_options, state=state)
    _validate_family_options(spec)
    return spec


def _validate_family_options(spec: "FunctionSpaceSpec") -> None:
    """Let the selected family own validation of its own parameters."""

    from neural_networks.function_spaces.factory import validate_function_space_options

    validate_function_space_options(spec)


@dataclass(frozen=True)
class FunctionSpaceSpec:
    """Immutable role-neutral family, options, and enabled state."""

    family: Optional[FunctionSpaceFamily]
    options: Mapping[str, Any]
    state: RoleState = RoleState.ENABLED

    def __post_init__(self) -> None:
        state = RoleState.from_value(self.state)
        object.__setattr__(self, "state", state)
        family = None if self.family is None else FunctionSpaceFamily.from_value(self.family)
        object.__setattr__(self, "family", family)
        if not isinstance(self.options, Mapping):
            raise ValueError(
                f"Function-space options must be a mapping, got {type(self.options).__name__}."
            )
        object.__setattr__(self, "options", _deep_freeze(self.options))
        if state is RoleState.DISABLED and family is not None:
            raise ValueError("A disabled function-space spec cannot have a family.")
        if state is RoleState.DISABLED and self.options:
            raise ValueError("A disabled function-space spec cannot have options.")
        if state is RoleState.ENABLED and family is None:
            raise ValueError("An enabled function-space spec requires a family.")

    @property
    def enabled(self) -> bool:
        return self.state is RoleState.ENABLED

    def __repr__(self) -> str:
        family = self.family.value if self.family is not None else None
        return (
            "FunctionSpaceSpec("
            f"family={family!r}, state={self.state.value!r}, "
            f"options={dict(self.options)!r})"
        )


@dataclass(frozen=True)
class ResolvedFunctionSpaceConfig:
    """The one resolved backend plus independent f and nuisance specs."""

    backend: TrainingBackend
    f: FunctionSpaceSpec
    nuisance: FunctionSpaceSpec

    def __repr__(self) -> str:
        return (
            "ResolvedFunctionSpaceConfig("
            f"backend={self.backend.value!r}, "
            f"f={self.f!r}, nuisance={self.nuisance!r})"
        )



def _coerce_role(value: FunctionSpaceSpec | Mapping[str, Any] | None, role: str) -> Optional[FunctionSpaceSpec]:
    if value is None:
        return None
    if isinstance(value, FunctionSpaceSpec):
        _validate_family_options(value)
        # Rebuild even immutable specs so f and nuisance never share an options
        # object when a caller intentionally supplies the same spec twice.
        return FunctionSpaceSpec(value.family, value.options, value.state)
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} function-space config must be a mapping or FunctionSpaceSpec.")
    return _role_spec_from_mapping(value, role)


def resolve_dual_role_config(
    *,
    backend: TrainingBackend | str | None = None,
    f: FunctionSpaceSpec | Mapping[str, Any] | None = None,
    nuisance: FunctionSpaceSpec | Mapping[str, Any] | None = None,
) -> ResolvedFunctionSpaceConfig:
    """Resolve the canonical function-space specifications for both roles."""
    resolved_backend = TrainingBackend.from_value(backend)
    resolved_f = _coerce_role(f, "f")
    resolved_nuisance = _coerce_role(nuisance, "nuisance")
    if resolved_f is None:
        raise ValueError("f function-space config is required.")
    if resolved_nuisance is None:
        raise ValueError("nuisance function-space config is required.")
    if resolved_f.state is RoleState.DISABLED:
        raise ValueError(
            "f role is disabled, but f may not be disabled; denominator omission remains model semantics."
        )
    if (
        resolved_backend is TrainingBackend.NPLM
        and (
            resolved_f.family is not FunctionSpaceFamily.ADAPTIVE_NEURAL
            or resolved_nuisance.enabled
            and resolved_nuisance.family is not FunctionSpaceFamily.BIN_INDICATORS
        )
    ):
        raise ValueError(
            "NPLM backend supports adaptive f and binned nuisance; "
            "select LFVDDP for the requested function-space families."
        )
    return ResolvedFunctionSpaceConfig(
        backend=resolved_backend,
        f=resolved_f,
        nuisance=resolved_nuisance,
    )



__all__ = [
    "FunctionSpaceFamily",
    "FunctionSpaceRole",
    "FunctionSpaceSpec",
    "ResolvedFunctionSpaceConfig",
    "RoleState",
    "TrainingBackend",
    "resolve_dual_role_config",
]
