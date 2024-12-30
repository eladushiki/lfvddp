from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from math import floor
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict

from data_tools.histogram_generation import exp, gauss, physics
from frame.config_handle import Config

from __future__ import ExpConfig, GaussConfig, PhysicsConfig

@dataclass
class ClusterConfig(Config):
    runtag: str

    # PERSONAL SSH parameters
    cluster__host_address: str
    cluster__user: str
    cluster__password: str

    # PERSONAL optional SSH jump for WIS remote work
    cluster__is_use_wsl_command_line: bool

    # PERSONAL run parameters
    cluster__remote_repository_dir: PurePosixPath
    cluster__working_dir: Path

    # qsub command parameters
    cluster__qsub_walltime: str  # in the form of "12:00:00"
    cluster__qsub_io: int
    cluster__qsub_mem: int
    cluster__qsub_cores: int

    # qstat command parameters
    cluster__qstat_n_jobs: int
    

@dataclass
class TrainConfig(ClusterConfig, ABC):
    # Train config subtype by name
    FUNCTION_CONFIGS_BY_NAME = {
        "exp": ExpConfig,
        "gauss": GaussConfig,
        "physics": PhysicsConfig,
    }

    # Data generation data locations
    train__data_dir: Path
    train__backgournd_distribution_path: Path
    train__signal_distribution_path: Path

    # Data set sizing
    train__batch_test_fraction: float  # as a fraction
    @property
    def train__batch_train_fraction(self):
        return 1 - self.train__batch_test_fraction
    
    # Data set types
    train__data_background_aux: str  # string of data sets in the auxillary background (e.g. 'Ref' or 'Sig+Bkg+Ref')
    train__data_signal_aux: str  # string of data sets in the auxillary sig (e.g. '' or 'Sig+Bkg')
    train__data_background: str  # string of data sets in the data background (e.g. 'Bkg' or 'Sig+Bkg')
    train__data_signal: str  # string of data sets in the data sig (e.g. 'Sig' or 'Sig+Bkg')

    # histogram settings
    train__histogram_is_binned: bool
    train__histogram_resolution: int
    train__histogram_is_use_analytic: int  # if 1: generate data from pdf
    train__histogram_analytic_pdf: str  # decides which samples to use (em or exp)

    # Resampling settings
    train__resample_is_resample: bool
    train__resample_label_method: str
    train__resample_method_type: str
    train__resample_is_replacement: bool

    # Signal parameters
    train__signal_number_of_events: int
    train__signal_types: str
    
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
    def analytic_background_function_calling_parameters(self) -> Dict[str, Any]:
        pass
    @property
    @abstractmethod
    def train__number_of_reference_events(self) -> int:
        pass
    @property
    @abstractmethod
    def train__number_of_background_events(self) -> int:
        pass


@dataclass
class ExpConfig(TrainConfig):
    @property
    def analytic_background_function(self) -> Callable:
        return exp
    @property
    def analytic_background_function_calling_parameters(self) -> Dict[str, Any]:
        return {
            "n_ref": self.train__number_of_reference_events,
            "n_bkg": self.train__number_of_background_events,
            "scale_factor": 1,
            "normalization_factor": 1,
            "signal_location": self.train_exp__signal_location,
            "signal_scale": self.train_exp__signal_scale,
            "poisson_fluctuations": self.train_exp__N_poiss,
            "is_resonant_signal_shape": self.train_exp__signal_resonant,
        }
    @property
    def train__number_of_reference_events(self) -> int:
        return round(219087 * self.train__batch_train_fraction * self.train__data_usage_fraction)
    @property
    def train__number_of_background_events(self) -> int:
        return round(219087 * self.train__batch_test_fraction * self.train__data_usage_fraction)
    
    train_exp__signal_location: int
    train_exp__signal_scale: float
    train_exp__N_poiss: int
    train_exp__signal_resonant: bool


@dataclass
class GaussConfig(TrainConfig):
    @property
    def analytic_background_function(self) -> Callable:
        return gauss
    @property
    def analytic_background_function_calling_parameters(self) -> Dict[str, Any]:
        return {
            "normalization_factor": 1,
            "signal_location": self.train_gauss__signal_location,
            "signal_scale": self.train_gauss__signal_scale,
            "poisson_fluctuations": self.train_gauss__N_poiss,
            "dim": 1,
        }
    @property
    def train__number_of_reference_events(self) -> int:
        return round(219087 * float(self.train__batch_train_fraction) * self.train__data_usage_fraction)
    @property
    def train__number_of_background_events(self) -> int:
        return round(219087 * float(Fraction(self.train__batch_test_fraction)) * self.train__data_usage_fraction)

    train_gauss__signal_location: int
    train_gauss__signal_scale: float
    train_gauss__N_poiss: int

@dataclass
class PhysicsConfig(TrainConfig):
    @property
    def analytic_background_function(self) -> Callable:
        return physics
    @property
    def analytic_background_function_calling_parameters(self) -> Dict[str, Any]:
        return {
            "channel": 'em',
            "signal_types": ["ggH_taue","vbfH_taue"],
            "used_physical_variables": ['Mcoll'],
            "poisson_fluctuations": self.train_physics__N_poiss,
            "combimed_portion": self.train_physics__data_usage_fraction,
            "binned": False,
            "resolution": 0.1,
        }
    @property
    def train__number_of_reference_events(self) -> int:
        return floor(self.train__batch_train_fraction)
    @property
    def train__number_of_background_events(self) -> int:
        return floor(self.train__batch_test_fraction)

    train_physics__N_poiss: int
    train_physics__data_usage_fraction: float  # As a fraction
