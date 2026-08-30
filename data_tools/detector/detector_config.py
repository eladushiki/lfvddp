from dataclasses import dataclass
from typing import List


@dataclass
class DetectorConfig:
    """Configuration for detector observable selection."""

    detector__detect_observable_names: List[str]

    # Detector effects are shared by datasets in each detector family.  Keep
    # these explicit so the configuration schema exposes exactly one A and one
    # B value for each effect kind.

    detector__effect_a_efficiency: str = ""
    detector__effect_a_efficiency_uncertainty: str = ""
    detector__effect_a_error: str = ""

    detector__effect_b_efficiency: str = ""
    detector__effect_b_efficiency_uncertainty: str = ""
    detector__effect_b_error: str = ""

    @property
    def detector__number_of_dimensions(self) -> int:
        return len(self.detector__detect_observable_names)
