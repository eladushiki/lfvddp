from logging import warning
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

    if config.train__epochs < 1e5 and config.train__like_NPLM or \
            config.train__epochs < 5e5 and not config.train__like_NPLM:
        warning("Training epochs not sufficient, train may not converge")
