from typing import Union

from data_tools.dataset_config import DatasetConfig
from frame.cluster.cluster_config import ClusterConfig
from frame.config_handle import UserConfig
from train.train_config import TrainConfig


def validate_configuration(config: Union[UserConfig, ClusterConfig, DatasetConfig, TrainConfig]):
    """
    Validate legality of configuration
    """
    # NN input has to equate to the number of dimensions in the dataset
    assert config.dataset__number_of_dimensions == config.train__nn_input_dimension
