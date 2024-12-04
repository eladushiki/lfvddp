from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional

from scipy import cluster
from frame.config_handle import Config

@dataclass
class ClusterConfig(Config):
    runtag: str

    # PERSONAL SSH parameters
    cluster__host_address: str
    cluster__user: str
    cluster__password: str

    # PERSONAL optional SSH jump for WIS remote work
    cluster__is_use_ssh_jump: bool
    cluster__jump_host_address: Optional[str]
    cluster__jump_user: Optional[str]
    cluster__jump_password: Optional[str]

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
class TrainConfig(ClusterConfig):
    # Data generation data locations
    train__data_dir: Path
    train__backgournd_distribution_path: Path
    train__signal_distribution_path: Path

    # Data set sizing
    train__batch_test_fraction: float  # as a fraction
    @property
    def train__batch_train_fraction(self):
        return 1 - self.train__batch_test_fraction
    train__data_usage_fraction: float  # as a fraction. total number of events (as a fraction) in the over-all sample (before splitting into A and B).

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

    # Still unclear parameters
    train__N_poiss: int
