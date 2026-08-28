from dataclasses import dataclass
from typing import List


@dataclass
class DetectorConfig:
    """Configuration for detector observable selection."""

    detector__detect_observable_names: List[str]

    @property
    def detector__number_of_dimensions(self) -> int:
        return len(self.detector__detect_observable_names)
