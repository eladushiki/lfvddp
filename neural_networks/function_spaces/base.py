"""Role-neutral contracts and metadata for shared function spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from train.function_space_config import FunctionSpaceFamily


@dataclass(frozen=True)
class FunctionSpaceMetadata:
    """Structural metadata consumed by later adapters and diagnostics."""

    regularity: str
    coefficient_topology: str


@runtime_checkable
class FunctionSpace(Protocol):
    """Role-neutral construction/evaluation contract.

    This protocol deliberately does not encode likelihood signs, SR/CR weighting,
    disabled roles, or training semantics.
    """

    family: FunctionSpaceFamily
    options: Mapping[str, Any]
    metadata: FunctionSpaceMetadata
    feature_count: int

    def features(self, events: Any) -> Any:
        ...

    def evaluate(self, events: Any) -> Any:
        ...
