from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Optional

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
    
    ## Data generation data locations
    train__data_dir: Path
    train__backgournd_distribution_path: Path
    train__signal_distribution_path: Path

    ## Data set sizing
    train__batch_test_fraction: float  # as a fraction
    @property
    def train__batch_train_fraction(self):
        return 1 - self.train__batch_test_fraction
    train__data_usage_fraction: float  # as a fraction, previously "combined portion"
    
    ## Generated data set composition definitions
    # Each string can be any of:
    # 'Ref' - reference data, represents the null (SM) hypothesis, or - when all nuisance parameters are 0
    # 'Bkg' - background data, later composing the "real experimental" data
    # 'Sig' - signal data, later added to the background to simulate new physics

    # We postulate that the auxiliary data has no expressions of new physics and that it contains
    # expression of the nuisance parameters. The dataset contains their statistics, and we
    # use it to measure them independently.
    # We generate an auxiliary dataset that contains the known physics with some disruption, to train the
    # net for the nuisace parameters.
    train__data_aux_background_composition: List[str]  # data sets in the auxillary background (e.g. ['Ref'] or ['Sig', 'Bkg', 'Ref'])
    train__data_aux_signal_composition: List[str]  # data sets in the auxillary signal (e.g. [] or ['Sig', 'Bkg'])
    
    # "Experimental" datasets resemble the to-be experimental data, though composed of
    # "fake" (generated) signal.
    # The dataset of interest (toy or not) is composed of these two.
    train__data_experimental_background_composition: List[str]  # data sets in the data background (e.g. ['Bkg'] or ['Sig', 'Bkg'])
    train__data_experimental_signal_composition: List[str]  # data sets in the data sig (e.g. ['Sig'] or ['Sig', 'Bkg'])

    ## Resampling settings
    train__resample_is_resample: bool
    train__resample_label_method: str
    train__resample_method_type: str
    train__resample_is_replacement: bool

    ## Signal parameters
    train__signal_number_of_events: int
    train__signal_resonant: bool
    train__signal_location: int
    train__signal_scale: float
    
    ## This is the defining attribute for the subclass
    train__histogram_analytic_pdf: str  # decides which samples to use (em or exp)

    # Other histogram settings
    train__histogram_is_binned: bool
    train__histogram_resolution: int
    train__histogram_is_use_analytic: int  # if 1: generate data from pdf

    ## Nuisance parameters
    # Correction - what should be taken into account about the nuisance parameters?
    # - "SHAPE" - both normalization and shape uncertainties are considered
    # - "NORM" - only normalization uncertainties are considered
    # - "" - systematic uncertainties are neglected (simple NPLM is run - no Delta calculation and Tau is calculated without nuisance parameters)
    train__nuisance_correction: str  # "SHAPE", "NORM" or "".

    train__nuisances_shape_sigma: float        # shape nuisance sigma  # todo: convert to a list to enable any number of those
    train__nuisances_shape_mean_sigmas: float       # shape nuisance reference, in terms of std
    train__nuisances_shape_reference_sigmas: float  # norm nuisance reference, in terms of std
    
    train__nuisances_norm_sigma: float        # norm nuisance sigma
    train__nuisances_norm_mean_sigmas: float       # in terms of std
    train__nuisances_norm_reference_sigmas: float  # in terms of std

    ## Timing parameters
    train__epochs: int
    train__patience: int

    ## NN parameters
    # Max for a single weight - a hyperparameter
    train__nn_weight_clipping: float
    # Architecture of the NN
    # composed of input and output dimensions, and the number of nodes in the inner layer
    train__nn_input_dimension: int
    train__nn_output_dimension: int
    train__nn_inner_layer_nodes: int
    @property
    def train__nn_architecture(self) -> List[int]:
        return [self.train__nn_input_dimension, self.train__nn_inner_layer_nodes, self.train__nn_output_dimension]
    @property
    def train__nn_degrees_of_freedom(self) -> int:
        return compute_df(
            input_size=self.train__nn_input_dimension,
            hidden_layers=1,
            output_size=self.train__nn_output_dimension,
        )
    train__nn_loss_function: str  # string before history/weights.h5 and .txt names (TAU or delta)

    ## Common properties with different implementations
    @property
    @abstractmethod
    def train__analytic_background_function(self) -> Callable:
        pass
    @property
    @abstractmethod
    def train__number_of_reference_events(self) -> int:
        pass
    @property
    @abstractmethod
    def train__number_of_background_events(self) -> int:
        pass

    ## Must have definition for dynamic class resolution
    @classmethod
    @abstractmethod
    def HISTOGRAM_NAME(cls) -> str:
        pass

    @classmethod
    def dynamic_class_resolve(cls, config_params: Dict[str, Any]):
        defining_attribute_name = "train__histogram_analytic_pdf"

        if defining_attribute_name not in config_params.keys():
            raise AttributeError(f"Missing defining attribute {defining_attribute_name} to resolve {cls} subclass")

        histogram_subtypes = cls.__subclasses__()
        for histogram_subtype in histogram_subtypes:
            if histogram_subtype.HISTOGRAM_NAME() == config_params[defining_attribute_name]:
                return histogram_subtype

        return cls
