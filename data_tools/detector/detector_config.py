from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DetectorConfig:
    """Configuration for detector observable selection."""

    detector__detect_observable_names: List[str]
    # Detector effects are shared by datasets in each detector family.
    detector__effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def effects_for_category(self, category: Any) -> Dict[str, Any]:
        family = getattr(category, "name", str(category)).split("_")[0].upper()
        return dict(self.detector__effects.get(family, {}))

    @property
    def detector__number_of_dimensions(self) -> int:
        return len(self.detector__detect_observable_names)
