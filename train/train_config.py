from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict

from frame.config_handle import Config

@dataclass
class ClusterConfig(Config):
    runtag: str

    # qsub command parameters
    cluster__qsub_walltime: str  # in the form of "12:00:00"
    cluster__qsub_io: int
    cluster__qsub_mem: int
    cluster__qsub_cores: int
    cluster__qsub_ngpus_for_train: int

    # qstat command parameters
    cluster__qstat_n_jobs: int
    

@dataclass
class TrainConfig(ClusterConfig, ABC):
    ## This is the defining attribute for the subclass
    train__histogram_analytic_pdf: str  # decides which samples to use (em or exp)

    # Other histogram settings
    train__histogram_is_binned: bool
    train__histogram_resolution: int
    train__histogram_is_use_analytic: int  # if 1: generate data from pdf
    
    # Data generation data locations
    train__data_dir: Path
    train__backgournd_distribution_path: Path
    train__signal_distribution_path: Path

    # Data set sizing
    train__batch_test_fraction: float  # as a fraction
    @property
    def train__batch_train_fraction(self):
        return 1 - self.train__batch_test_fraction
    train__data_usage_fraction: float  # as a fraction
    
    # Data set types
    train__data_background_aux: str  # string of data sets in the auxillary background (e.g. 'Ref' or 'Sig+Bkg+Ref')
    train__data_signal_aux: str  # string of data sets in the auxillary sig (e.g. '' or 'Sig+Bkg')
    train__data_background: str  # string of data sets in the data background (e.g. 'Bkg' or 'Sig+Bkg')
    train__data_signal: str  # string of data sets in the data sig (e.g. 'Sig' or 'Sig+Bkg')

    # Resampling settings
    train__resample_is_resample: bool
    train__resample_label_method: str
    train__resample_method_type: str
    train__resample_is_replacement: bool

    # Signal parameters
    train__signal_number_of_events: int
    train__signal_types: str
    train__signal_resonant: bool
    train__signal_location: int
    train__signal_scale: float
    
    # Nuisance parameters
    train__nuisance_correction: str # "SHAPE" or "NORM"
    train__nuisance_scale: str      # shape nuisance reference
    train__nuisance_norm: str       # norm nuisance reference
    train__nuisance_sigma_s: str    # shape nuisance sigma
    train__nuisance_sigma_n: str    # norm nuisance sigma

    # Timing parameters
    train__epochs_type: str  # "TAU" or "delta"
    train__epochs: int
    train__patience: int

    # NN parameters
    train__nn_weight_clipping: float
    train__nn_architecture: str  # "1:4:1", first digit is dimension - 1 = 1d, 2 = table = 2d
    train__nn_input_size: int
    train__nn_loss_function: str  # string before history/weights.h5 and .txt names (TAU or delta)

    # Common properties with different implementations
    @property
    @abstractmethod
    def analytic_background_function(self) -> Callable:
        pass
    @property
    @abstractmethod
    def train__number_of_reference_events(self) -> int:
        pass
    @property
    @abstractmethod
    def train__number_of_background_events(self) -> int:
        pass

    # Must have definition for dynamic class resolution
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
