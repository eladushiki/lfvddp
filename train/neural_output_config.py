"""Configuration for the bounded output of LFVDDP neural function spaces."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

LEGACY_NEURAL_OUTPUT_MAPPING = "tanh"


@dataclass(frozen=True)
class NeuralOutputConfig:
    output_mapping: str = "algebraic"

    def __post_init__(self) -> None:
        if self.output_mapping not in ("algebraic", LEGACY_NEURAL_OUTPUT_MAPPING):
            raise ValueError("output_mapping must be 'algebraic' or 'tanh'.")

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "NeuralOutputConfig":
        if "output_mapping" in options:
            return cls(output_mapping=options["output_mapping"])
        return cls()
