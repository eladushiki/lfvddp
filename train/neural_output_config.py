"""Configuration for the bounded output of LFVDDP neural function spaces."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from frame.value_enum import ValueEnum


class NeuralOutputMapping(ValueEnum):
    """Supported bounded output maps, with their serialized configuration names."""

    ALGEBRAIC = "algebraic"
    TANH = "tanh"


@dataclass(frozen=True)
class NeuralOutputConfig:
    output_mapping: NeuralOutputMapping | str = NeuralOutputMapping.ALGEBRAIC

    def __post_init__(self) -> None:
        try:
            mapping = NeuralOutputMapping.parse(self.output_mapping)
        except ValueError as error:
            raise ValueError(f"Invalid output_mapping: {error}") from error
        object.__setattr__(self, "output_mapping", mapping)

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "NeuralOutputConfig":
        if "output_mapping" in options:
            return cls(output_mapping=options["output_mapping"])
        return cls()
