from data_tools.dataset_config import (
    DatasetConfig,
    GeneratedDatasetParameters,
    LoadedDatasetParameters,
)
from data_tools.detector.detector_config import DetectorConfig
from frame.cluster.cluster_config import ClusterConfig
from frame.config_handle import UserConfig
from plot.plotting_config import PlottingConfig
from train.train_config import TrainConfig


def cross_configure(
    config: ClusterConfig
    | DatasetConfig
    | DetectorConfig
    | PlottingConfig
    | TrainConfig
    | UserConfig,
) -> None:
    """Fill defaults that depend on the fully merged configuration."""
    detector_dimension = config.detector__number_of_dimensions
    generated_type = GeneratedDatasetParameters.DATASET_PARAMETER_TYPE_NAME()
    loaded_type = LoadedDatasetParameters.DATASET_PARAMETER_TYPE_NAME()
    for dataset_definition in config.dataset__definitions:
        dataset_type = dataset_definition.get(config._dataset__type_property)
        if dataset_type == generated_type:
            dataset_definition.setdefault(
                "dataset_generated__number_of_dimensions",
                detector_dimension,
            )
        elif (
            dataset_type == loaded_type
            and dataset_definition.get("dataset__signal_generator") is not None
        ):
            dataset_definition.setdefault(
                "dataset_loaded__signal_observable_names",
                list(config.detector__detect_observable_names),
            )


def cross_validate(
    config: ClusterConfig
    | DatasetConfig
    | DetectorConfig
    | PlottingConfig
    | TrainConfig
    | UserConfig,
):
    if config.cluster__qsub_needs_continuation and config.train__is_nplm:
        raise NotImplementedError(
            "Long-walltime continuation is only implemented for LFVNN/PyTorch training."
        )

    if config.train__final_learning_rate is not None:
        assert config.train__final_learning_rate <= config.train__learning_rate, (
            "Final learning rate must not exceed the initial learning rate."
        )
