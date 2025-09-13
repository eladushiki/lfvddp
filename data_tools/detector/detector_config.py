from dataclasses import dataclass, field
from typing import List


@dataclass
class DetectorConfig:
    detector__number_of_dimensions: int
    detector__detect_observable_names: List[str]
    detector__binning_maxima: List[int]

    # Default enabled parameters
    detector__binning_minima: List[int] = field(default=0)
    detector__binning_number_of_bins: List[int] = field(default=30)
    
    def __post_init__(self):
        # Detector dimensions should fit DataSet dimension. Inserts default and expands dimensions if given an int.
        if isinstance(self.detector__binning_minima, int):
            self.detector__binning_minima = [self.detector__binning_minima] * self.detector__number_of_dimensions

        if isinstance(self.detector__binning_number_of_bins, int):
            self.detector__binning_number_of_bins = [self.detector__binning_number_of_bins] * self.detector__number_of_dimensions

        self.validate()
    
    def validate(self):
        assert len(self.detector__binning_minima) == self.detector__number_of_dimensions, \
            f"Detector binning minima length {len(self.detector__binning_minima)} does not match "\
            f"Detector number of dimensions {self.detector__number_of_dimensions}"

        assert len(self.detector__binning_maxima) == self.detector__number_of_dimensions, \
            f"Detector binning maxima length {len(self.detector__binning_maxima)} does not match "\
            f"Detector number of dimensions {self.detector__number_of_dimensions}"

        assert len(self.detector__binning_number_of_bins) == self.detector__number_of_dimensions, \
            f"Detector binning number of bins length {len(self.detector__binning_number_of_bins)} does not match "\
            f"Detector number of dimensions {self.detector__number_of_dimensions}"
