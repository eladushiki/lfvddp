"""Shared string-enum behavior for configuration and metadata values."""

from enum import Enum


class ValueEnum(str, Enum):
    """String enum with useful configuration-file coercion semantics."""

    def __str__(self) -> str:
        return self.value
