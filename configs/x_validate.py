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
    assert config.train__nn_input_dimension == config.detector__number_of_dimensions, \
        f"Input dimension {config.train__nn_input_dimension} does not match detector dimension " \
        f"{config.detector__number_of_dimensions}"
