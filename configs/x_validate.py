from typing import Union

from data_tools.dataset_config import DatasetConfig, GeneratedDatasetParameters
from data_tools.detector.detector_config import DetectorConfig
from frame.cluster.cluster_config import ClusterConfig
from frame.config_handle import UserConfig
from plot.plotting_config import PlottingConfig
from train.train_config import TrainConfig


def cross_configure(config: Union[
    ClusterConfig,
    DatasetConfig,
    DetectorConfig,
    PlottingConfig,
    TrainConfig,
    UserConfig,
]) -> None:
    """Fill defaults that depend on the fully merged configuration."""
    detector_dimension = config.detector__number_of_dimensions

    if config.train__nn_input_dimension is None:
        config.train__nn_input_dimension = detector_dimension

    generated_type = GeneratedDatasetParameters.DATASET_PARAMETER_TYPE_NAME()
    for dataset_definition in config.dataset__definitions:
        if dataset_definition.get(config._dataset__type_property) == generated_type:
            dataset_definition.setdefault(
                "dataset_generated__number_of_dimensions",
                detector_dimension,
            )


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

    if config.cluster__qsub_needs_continuation and config.train__like_NPLM:
        raise NotImplementedError("Long-walltime continuation is only implemented for LFVNN/PyTorch training.")

    assert config.train__nn_input_dimension == config.detector__number_of_dimensions, \
        f"Input dimension {config.train__nn_input_dimension} does not match detector dimension " \
        f"{config.detector__number_of_dimensions}"
