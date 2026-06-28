from typing import Union
from data_tools.dataset_config import DatasetConfig
from data_tools.detector.detector_config import DetectorConfig
from frame.cluster.cluster_config import ClusterConfig
from frame.config_handle import UserConfig
from plot.plotting_config import PlottingConfig
from train.train_config import TrainConfig


def cross_validate(config: Union[
    ClusterConfig,
    DatasetConfig,
    DetectorConfig,
    PlottingConfig,
    TrainConfig,
    UserConfig,
]):
    # Resolve qsub CPU allocation after merged config is fully constructed.
    if config.cluster__qsub_ncpus is None:
        config.cluster__qsub_ncpus = 2 if config.train__run_symmetric_in_parallel else 1

    if config.cluster__qsub_needs_continuation and config.train__like_NPLM:
        raise NotImplementedError("Long-walltime continuation is only implemented for LFVNN/PyTorch training.")

    assert config.train__nn_input_dimension == config.detector__number_of_dimensions, \
        f"Input dimension {config.train__nn_input_dimension} does not match detector dimension " \
        f"{config.detector__number_of_dimensions}"
