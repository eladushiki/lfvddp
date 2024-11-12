from dataclasses import dataclass
from frame.config_handle import Config

@dataclass
class ClusterConfig(Config):
    n_jobs: int
    runtag: str
    walltime: str  # in the form of "12:00:00"
    save_walltime: str  # in the form of "05:00"
    remove: bool

@dataclass
class TrainConfig(ClusterConfig):
    # Data set sizing
    train__batch_size: str  # as a fraction
    train__test_batch_size: str  # as a fraction
    train__combined_portion: float  # as a fraction

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
    train__nn_architecture: str  # "1:4:1"
    train__nn_input_size: int
    train__nn_loss_function: str  # string before history/weights.h5 and .txt names (TAU or delta)
