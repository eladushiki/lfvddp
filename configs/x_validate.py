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
        has_parallel_training = (
            config.train__run_symmetric_in_parallel
            if isinstance(config, TrainConfig)
            else False
        )
        config.cluster__qsub_ncpus = 2 if has_parallel_training else 1

    assert config.train__nn_input_dimension == config.detector__number_of_dimensions, \
        f"Input dimension {config.train__nn_input_dimension} does not match detector dimension " \
        f"{config.detector__number_of_dimensions}"
