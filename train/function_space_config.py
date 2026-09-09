"""Canonical configuration for the two learned likelihood roles.

This module deliberately contains no model or evaluator code.  It translates the
flat configuration used by older config packs into one immutable description
that both role adapters can consume.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Optional


class _ValueEnum(str, Enum):
    """String enum with useful config-file coercion semantics."""

    def __str__(self) -> str:
        return self.value


class TrainingBackend(_ValueEnum):
    """Training implementation, orthogonal to mathematical function family."""

    LFVDDP = "lfvddp"
    # LFVNN was the historical name used in a few explanatory documents.
    LFVNN = "lfvddp"
    NPLM = "nplm"

    @classmethod
    def from_value(cls, value: TrainingBackend | str | None) -> TrainingBackend:
        if value is None:
            return cls.LFVDDP
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {"lfvnn": cls.LFVDDP, "lfvddp": cls.LFVDDP, "nplm": cls.NPLM}
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(
                f"Unknown training backend {value!r}; expected one of: lfvddp, nplm."
            ) from error


class FunctionSpaceFamily(_ValueEnum):
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
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "adaptive": cls.ADAPTIVE_NEURAL,
            "adaptive_neural_network": cls.ADAPTIVE_NEURAL,
            "bin_indicator": cls.BIN_INDICATORS,
            "bins": cls.BIN_INDICATORS,
            "gaussian_rbf": cls.GAUSSIAN_RADIAL_BASIS,
            "rbf": cls.GAUSSIAN_RADIAL_BASIS,
        }
        try:
            if normalized in aliases:
                return aliases[normalized]
            return cls(normalized)
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown function-space family {value!r}; expected one of: {choices}."
            ) from error


class RoleState(_ValueEnum):
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
        if isinstance(value, bool):
            return cls.ENABLED if value else cls.DISABLED
        normalized = str(value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as error:
            raise ValueError(
                f"Unknown role state {value!r}; expected 'enabled' or 'disabled'."
            ) from error


_SENSITIVE_KEY_PARTS = ("password", "secret", "token", "credential", "private_key")


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


def _sanitized(value: Any, *, key: str = "") -> Any:
    """Return a bounded, non-secret representation for diagnostics."""
    if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(name): _sanitized(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        values = list(value)
        if len(values) > 8:
            return [_sanitized(item) for item in values[:8]] + [f"<... {len(values) - 8} more>"]
        return [_sanitized(item) for item in values]
    if isinstance(value, str) and len(value) > 120:
        return value[:117] + "..."
    return value


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
    _validate_family_options(spec, role)
    return spec


def _geometry_rows(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    """Normalize scalar/flat/nested numeric config values for validation."""
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be numeric geometry.")
    if isinstance(value, Real):
        rows = ((float(value),),)
    else:
        try:
            values = tuple(value)
        except TypeError as error:
            raise ValueError(f"{name} must be numeric geometry.") from error
        if not values:
            raise ValueError(f"{name} must not be empty.")
        if all(isinstance(item, Real) and not isinstance(item, bool) for item in values):
            rows = (tuple(float(item) for item in values),)
        else:
            rows = tuple(_geometry_rows(item, f"{name}[{index}]")[0] for index, item in enumerate(values))
    if any(not row or any(not math.isfinite(item) for item in row) for row in rows):
        raise ValueError(f"{name} must contain finite numeric values.")
    return rows


def _validate_family_options(spec: "FunctionSpaceSpec", role: str) -> None:
    if spec.state is RoleState.DISABLED:
        return
    assert spec.family is not None
    options = spec.options
    required: dict[FunctionSpaceFamily, tuple[str, ...]] = {
        FunctionSpaceFamily.BIN_INDICATORS: ("minima", "maxima", "number_of_bins"),
        FunctionSpaceFamily.CUBIC_BSPLINE: ("knots",),
        FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL: ("basis", "maximum_degree", "domain"),
        FunctionSpaceFamily.FIXED_SIGMOID: ("centers", "widths"),
        FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS: ("centers", "widths"),
    }
    missing = [name for name in required.get(spec.family, ()) if name not in options]
    if missing:
        raise ValueError(
            f"{role} family {spec.family.value!r} requires option(s): {', '.join(missing)}."
        )
    if spec.family is FunctionSpaceFamily.CUBIC_BSPLINE:
        knot_rows = _geometry_rows(options["knots"], f"{role}.options.knots")
        for knots in knot_rows:
            if len(knots) < 2 or any(left >= right for left, right in zip(knots, knots[1:])):
                # Full clamped vectors are allowed, but still need a valid span.
                if len(knots) < 8 or knots[0] >= knots[-1] or any(left > right for left, right in zip(knots, knots[1:])):
                    raise ValueError(f"{role}.options.knots must be increasing spline knots.")
    elif spec.family is FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL:
        basis = str(options["basis"]).strip().lower()
        if basis not in {"legendre", "chebyshev"}:
            raise ValueError(
                f"{role}.options.basis must be 'legendre' or 'chebyshev', got {basis!r}."
            )
        degree = options["maximum_degree"]
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValueError(f"{role}.options.maximum_degree must be a nonnegative integer.")
        domains = _geometry_rows(options["domain"], f"{role}.options.domain")
        if any(len(domain) != 2 or domain[0] >= domain[1] for domain in domains):
            raise ValueError(f"{role}.options.domain must contain increasing (minimum, maximum) pairs.")
    elif spec.family in {FunctionSpaceFamily.FIXED_SIGMOID, FunctionSpaceFamily.GAUSSIAN_RADIAL_BASIS}:
        centers = _geometry_rows(options["centers"], f"{role}.options.centers")
        widths = _geometry_rows(options["widths"], f"{role}.options.widths")
        if any(width <= 0 for row in widths for width in row):
            raise ValueError(f"{role}.options.widths must be strictly positive.")
        if len(widths) > 1 and len(widths) not in {len(centers), len(centers[0])}:
            raise ValueError(f"{role}.options.widths has incompatible geometry.")


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
            f"options={_sanitized(self.options)!r})"
        )


@dataclass(frozen=True)
class ResolvedFunctionSpaceConfig:
    """The one resolved backend plus independent f and nuisance specs."""

    backend: TrainingBackend
    f: FunctionSpaceSpec
    nuisance: FunctionSpaceSpec
    compatibility_source: str = "canonical"

    @property
    def f_spec(self) -> FunctionSpaceSpec:
        return self.f

    @property
    def nuisance_spec(self) -> FunctionSpaceSpec:
        return self.nuisance

    def __repr__(self) -> str:
        return (
            "ResolvedFunctionSpaceConfig("
            f"backend={self.backend.value!r}, compatibility_source={self.compatibility_source!r}, "
            f"f={self.f!r}, nuisance={self.nuisance!r})"
        )


# Descriptive aliases for callers that prefer the domain term over "resolved".
DualRoleFunctionSpaceConfig = ResolvedFunctionSpaceConfig
ResolvedDualRoleConfig = ResolvedFunctionSpaceConfig


def _coerce_role(value: FunctionSpaceSpec | Mapping[str, Any] | None, role: str) -> Optional[FunctionSpaceSpec]:
    if value is None:
        return None
    if isinstance(value, FunctionSpaceSpec):
        _validate_family_options(value, role)
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
    legacy_f_options: Optional[Mapping[str, Any]] = None,
    legacy_nuisance_options: Optional[Mapping[str, Any]] = None,
    legacy_nuisance_enabled: bool = True,
    legacy_like_nplm: bool = False,
    compatibility_source: Optional[str] = None,
    validate_legacy: bool = True,
) -> ResolvedFunctionSpaceConfig:
    """Resolve canonical role objects, or translate the legacy flat fields once."""
    resolved_backend = TrainingBackend.from_value(backend)
    if legacy_like_nplm and resolved_backend is TrainingBackend.LFVDDP:
        if backend is not None:
            raise ValueError(
                "train__backend='lfvddp' conflicts with legacy train__like_NPLM=True."
            )
        resolved_backend = TrainingBackend.NPLM

    legacy_f_options = dict(legacy_f_options or {})
    legacy_nuisance_options = dict(legacy_nuisance_options or {})
    resolved_f = _coerce_role(f, "f")
    resolved_nuisance = _coerce_role(nuisance, "nuisance")
    used_legacy = False

    if resolved_f is None:
        used_legacy = True
        resolved_f = FunctionSpaceSpec(
            family=FunctionSpaceFamily.ADAPTIVE_NEURAL,
            options=legacy_f_options,
        )
        _validate_family_options(resolved_f, "f")
    if resolved_f is not None and resolved_f.state is RoleState.DISABLED:
        raise ValueError(
            "f role is disabled, but f may not be disabled; denominator omission remains model semantics."
        )

    if resolved_nuisance is None:
        used_legacy = True
        if not legacy_nuisance_enabled:
            resolved_nuisance = FunctionSpaceSpec(
                family=None, options={}, state=RoleState.DISABLED
            )
        else:
            # Legacy nuisance defaults to the existing scalar binned estimator.
            resolved_nuisance = FunctionSpaceSpec(
                family=FunctionSpaceFamily.BIN_INDICATORS,
                options=legacy_nuisance_options,
            )
            if validate_legacy:
                _validate_family_options(resolved_nuisance, "nuisance")

    if resolved_backend is TrainingBackend.NPLM and (
        resolved_f.family is not FunctionSpaceFamily.ADAPTIVE_NEURAL
        or resolved_nuisance.enabled
        and resolved_nuisance.family is not FunctionSpaceFamily.BIN_INDICATORS
    ):
        raise ValueError(
            "NPLM backend currently supports only legacy adaptive f and binned nuisance; "
            "select LFVDDP for the requested function-space families."
        )

    source = compatibility_source or ("legacy" if used_legacy else "canonical")
    return ResolvedFunctionSpaceConfig(
        backend=resolved_backend,
        f=resolved_f,
        nuisance=resolved_nuisance,
        compatibility_source=source,
    )


# Public spelling used by configuration callers and future adapters.
def resolve_function_space_config(**kwargs: Any) -> ResolvedFunctionSpaceConfig:
    return resolve_dual_role_config(**kwargs)


__all__ = [
    "DualRoleFunctionSpaceConfig",
    "FunctionSpaceFamily",
    "FunctionSpaceSpec",
    "ResolvedDualRoleConfig",
    "ResolvedFunctionSpaceConfig",
    "RoleState",
    "TrainingBackend",
    "resolve_dual_role_config",
    "resolve_function_space_config",
]
