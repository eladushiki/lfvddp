from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Callable, Dict, List

from frame.config_handle import Config
from neural_networks.NPLM.src.NPLM.PLOTutils import compute_df

@dataclass
class ClusterConfig(Config, ABC):
    runtag: str
    cluster__project_root_at_cluster_abspath: PurePosixPath
    cluster__environment_activation_script_at_cluster_abspath: PurePosixPath

    # qsub command parameters
    cluster__qsub_queue: str
    cluster__qsub_n_jobs: int
    cluster__qsub_job_name: str
    cluster__qsub_walltime: str  # in the form of "12:00:00"
    cluster__qsub_io: int
    cluster__qsub_mem: int
    cluster__qsub_ngpus_for_train: int
    

@dataclass
class TrainConfig(ClusterConfig, ABC):

    ## Using real dataset parts implementation are left for the reader.
    # said reader would like to, firsly, implement
    # train__histogram_is_use_analytic: int  # if 1: generate data from pdf
    
    ## Data set composition definitions
    # Each string of dataset composition can be any of the following parts:
    # 'Ref' - reference data, represents the null (SM) hypothesis, or - when all nuisance parameters are 0
    # 'Bkg' - background data, later composing the "real experimental" data
    # 'Sig' - signal data, later added to the background to simulate new physics

    # For 'Ref' and 'Bkg'
    train__background_data_generation_function: str  # This is the defining attribute for the subclass
    @property
    @abstractmethod
    def train__analytic_background_function(self) -> Callable:
        pass

    # Reference 'Ref' parameters
    train__number_of_reference_events: int

    # Background 'Bkg' parameters
    train__number_of_background_events: int

    # Signal 'Sig' parameters
    train__number_of_signal_events: int
    train__signal_location: int

    ## Training for nuisance parameters
    # Correction - what should be taken into account about the nuisance parameters?
    # - "SHAPE" - both normalization and shape uncertainties are considered
    # - "NORM" - only normalization uncertainties are considered
    # - "" - systematic uncertainties are neglected (simple NPLM is run - no Delta calculation and Tau is calculated without nuisance parameters)
    train__nuisance_correction: str  # "SHAPE", "NORM" or "".
    @property
    def train__data_is_train_for_nuisances(self) -> bool:
        return self.train__nuisance_correction != ""

    train__nuisances_shape_sigma: float        # shape nuisance sigma  # todo: convert to a list to enable any number of those
    train__nuisances_shape_mean_sigmas: float       # shape nuisance reference, in terms of std
    train__nuisances_shape_reference_sigmas: float  # norm nuisance reference, in terms of std
    
    train__nuisances_norm_sigma: float        # norm nuisance sigma
    train__nuisances_norm_mean_sigmas: float       # in terms of std
    train__nuisances_norm_reference_sigmas: float  # in terms of std

    ## We postulate that the auxiliary data has no expressions of new physics and that it contains
    # expression of the nuisance parameters. The dataset contains their statistics, and we
    # use it to measure them independently.
    # We generate an auxiliary dataset that contains the known physics with some disruption, to train the
    # net for the nuisace parameters.
    train__data_aux_background_composition: List[str]
    train__data_aux_signal_composition: List[str]
    @property
    def train__data_aux_composition(self) -> List[str]:
        return self.train__data_aux_background_composition + self.train__data_aux_signal_composition
    
    # "Experimental" datasets resemble the to-be experimental data, though composed of
    # "fake" (generated) signal.
    # The flavor violation comparison is checked between these two.
    train__dataset_A_background_composition: List[str]
    train__dataset_A_signal_composition: List[str]
    @property
    def train__dataset_A_composition(self) -> List[str]:
        return self.train__dataset_A_background_composition + self.train__dataset_A_signal_composition
    
    train__dataset_B_background_composition: List[str]
    train__dataset_B_signal_composition: List[str]
    @property
    def train__dataset_B_composition(self) -> List[str]:
        return self.train__dataset_B_background_composition + self.train__dataset_B_signal_composition

    ## Resampling settings
    train__resample_is_resample: bool
    train__resample_label_method: str
    train__resample_method_type: str
    train__resample_is_replacement: bool

    ## Training parameters
    train__epochs: int
    train__number_of_epochs_for_checkpoint: int

    # NN parameters
    train__nn_weight_clipping: float

    train__nn_input_dimension: int
    train__nn_inner_layer_nodes: int
    train__nn_output_dimension: int
    @property
    def train__nn_architecture(self) -> List[int]:
        return [self.train__nn_input_dimension, self.train__nn_inner_layer_nodes, self.train__nn_output_dimension]
    @property
    def train__nn_degrees_of_freedom(self) -> int:
        return compute_df(
            input_size=self.train__nn_input_dimension,
            hidden_layers=self.train__nn_architecture[1:-1],
            output_size=self.train__nn_output_dimension,
        ) - 1  # The substraction is due to the argument about another constraint on the DoF in our paper

    ## Must have definition for dynamic class resolution
    @classmethod
    @abstractmethod
    def HISTOGRAM_NAME(cls) -> str:
        pass

    @classmethod
    def dynamic_class_resolve(cls, config_params: Dict[str, Any]):
        defining_attribute_name = "train__background_data_generation_function"

        if defining_attribute_name not in config_params.keys():
            raise AttributeError(f"Missing defining attribute {defining_attribute_name} to resolve {cls} subclass")

        histogram_subtypes = cls.__subclasses__()
        for histogram_subtype in histogram_subtypes:
            if histogram_subtype.HISTOGRAM_NAME() == config_params[defining_attribute_name]:
                return histogram_subtype

        return cls
