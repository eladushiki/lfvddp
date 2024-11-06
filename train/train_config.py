from random import sample
import test
from frame.config_handle import Config


class ClusterConfig(Config):
    n_jobs: int
    runtag: str
    train_size: str  # as a fraction
    test_size: str  # as a fraction
    number_of_signals: int
    walltime: str  # in the form of "12:00:00"
    save_walltime: str  # in the form of "05:00"
    remove: bool
    sample: str  # "exp" or ??
    pdf: int


class TrainConfig(ClusterConfig):
    # todo: some of these, say bools, are sometimes put as strings ("False")
    ch: str
    vars: str
    sig_types: str
    norm: int
    epochs: int
    patience: int
    run: int
    architecture: str  # "1:4:1"
    NPLM: bool
    WC: int
    Sig_loc: int
    Sig_scale: float
    pois: bool
    Loss: str  # "imperfect"
    validation: int
    resonant: bool
    CP: int
    resample: bool
    label_method: str
    N_method: str  # "fixed"
    replacement: bool
    original_seed: int
    binned: bool
    resolution: int
    NR: bool
    ND: bool
