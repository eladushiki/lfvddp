"""Shared string-enum behavior for configuration and metadata values."""

from enum import Enum


class ValueEnum(str, Enum):
    """String enum with useful configuration-file coercion semantics."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def parse(cls, value):
        """Return one member from a case-insensitive configuration value."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            choices = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown {cls.__name__} {value!r}; expected one of: {choices}."
            ) from error
