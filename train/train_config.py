from frame.config_handle import Config


class ClusterConfig(Config):
    njobs: int


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
