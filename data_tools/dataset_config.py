from abc import abstractmethod
from typing import Callable, List
from frame.cluster.cluster_config import ClusterConfig


class DatasetConfig(ClusterConfig):
    
    ## Using real dataset parts implementation are left for the reader.
    # said reader would like to, firsly, implement
    # train__histogram_is_use_analytic: int  # if 1: generate data from pdf
    
    ## Data set composition definitions
    # Each string of dataset composition can be any of the following parts:
    # 'Ref' - reference data, represents the null (SM) hypothesis, or - when all nuisance parameters are 0
    # 'Bkg' - background data, later composing the "real experimental" data
    # 'Sig' - signal data, later added to the background to simulate new physics

    # For 'Ref' and 'Bkg'
    dataset__background_data_generation_function: str  # This is the defining attribute for the subclass
    @property
    @abstractmethod
    def dataset__analytic_background_function(self) -> Callable:
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
    
    dataset__dataset_B_background_composition: List[str]
    dataset__dataset_B_signal_composition: List[str]
    @property
    def dataset__dataset_B_composition(self) -> List[str]:
        return self.dataset__dataset_B_background_composition + self.dataset__dataset_B_signal_composition

    ## Resampling settings
    dataset__resample_is_resample: bool
    dataset__resample_label_method: str
    dataset__resample_method_type: str
    dataset__resample_is_replacement: bool
