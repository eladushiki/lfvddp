from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import numpy.typing as npt


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

    def observable_bins(self, observable_name: str) -> Tuple[npt.NDArray, npt.NDArray]:
        try:
            index = self.detector__detect_observable_names.index(observable_name)
        except ValueError:
            raise ValueError(f"Observable name {observable_name} not found in detector observable names {self.detector__detect_observable_names}")
        
        bins_edges = np.linspace(
            self.detector__binning_minima[index],
            self.detector__binning_maxima[index],
            self.detector__binning_number_of_bins[index],
        )
        bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
        return bins_edges, bin_centers
