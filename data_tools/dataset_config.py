from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class DatasetConfig(ABC):
    
    ## Using real dataset parts implementation are left for the reader.
    # said reader would like to, firsly, implement
    # train__histogram_is_use_analytic: int  # if 1: generate data from pdf
    
    ## Utility dataset composition definitions - they later compose the used ones.
    # Each string of dataset composition can be any of the following parts:
    # 'Ref' - reference data, represents the null (SM) hypothesis, or - when all nuisance parameters are 0
    # 'Bkg' - background data, later composing the "real experimental" data
    # 'Sig' - signal data, later added to the background to simulate new physics

    # For 'Ref' and 'Bkg'
    dataset__background_data_generation_function: str  # This is the defining attribute for the subclass
    @property
    @abstractmethod
    def dataset__analytic_background_function(self) -> Callable[[DatasetConfig], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        pass

    # Reference 'Ref' parameters
    dataset__number_of_reference_events: int

    # Background 'Bkg' parameters
    dataset__number_of_background_events: int

    # Signal 'Sig' parameters
    dataset__number_of_signal_events: int
    dataset__signal_location: int

    ## We postulate that the auxiliary data has no expressions of new physics and that it contains
    # expression of the nuisance parameters. The dataset contains their statistics, and we
    # use it to measure them independently.
    # We generate an auxiliary dataset that contains the known physics with some disruption, to train the
    # net for the nuisace parameters.

    dataset__data_aux_background_composition: List[str]
    dataset__data_aux_signal_composition: List[str]
    @property
    def dataset__data_aux_composition(self) -> List[str]:
        return self.dataset__data_aux_background_composition + self.dataset__data_aux_signal_composition
    
    # "Experimental" datasets resemble the to-be experimental data, though composed of
    # "fake" (generated) signal.
    # The flavor violation comparison is checked between these two.
    dataset__dataset_A_background_composition: List[str]
    dataset__dataset_A_signal_composition: List[str]
    @property
    def dataset__dataset_A_composition(self) -> List[str]:
        return self.dataset__dataset_A_background_composition + self.dataset__dataset_A_signal_composition
    dataset__dataset_A_detector_efficiency: str
    dataset__dataset_A_detector_error: str

    dataset__dataset_B_background_composition: List[str]
    dataset__dataset_B_signal_composition: List[str]
    @property
    def dataset__dataset_B_composition(self) -> List[str]:
        return self.dataset__dataset_B_background_composition + self.dataset__dataset_B_signal_composition
    dataset__dataset_B_detector_efficiency: str
    dataset__dataset_B_detector_error: str

    dataset__nuisances_shape_sigma: float             # shape nuisance sigma  # todo: convert to a list to enable any number of those
    dataset__nuisances_shape_mean_sigmas: float       # shape nuisance reference, in terms of std
    dataset__nuisances_shape_reference_sigmas: float  # norm nuisance reference, in terms of std
    
    dataset__nuisances_norm_sigma: float              # norm nuisance sigma
    dataset__nuisances_norm_mean_sigmas: float        # in terms of std
    dataset__nuisances_norm_reference_sigmas: float   # in terms of std

    ## Resampling settings
    dataset__resample_is_resample: bool
    dataset__resample_label_method: str
    dataset__resample_method_type: str
    dataset__resample_is_replacement: bool
