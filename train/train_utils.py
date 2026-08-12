from typing import Union
from data_tools.detector.detector_config import DetectorConfig
from train.train_config import TrainConfig


# This is a bad location for this, but it is a workaround for the fact that
# these bits of data are not available anywhere else and could not be unified.
def model_degrees_of_freedom(
        config: Union[TrainConfig, DetectorConfig],
        is_numerator: bool,
) -> int:
    """Return the number of fitted parameters for one test-statistic model."""
    if not isinstance(config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    if not isinstance(config, DetectorConfig):
        raise TypeError(f"Expected DetectorConfig, got {config.__class__.__name__}")

    if config.train__like_NPLM:
        if not is_numerator:
            return 0

        from neural_networks.NPLM.src.NPLM.PLOTutils import compute_df

        return compute_df(
            input_size=config.train__nn_input_dimension,
            hidden_layers=[config.train__nn_inner_layer_nodes],
            output_size=config.train__nn_output_dimension,
        )

    nuisance_degrees_of_freedom = (
        config.detector__number_of_nuisance_parameters
        if config.train__data_is_train_for_nuisances
        else 0
    )
    if not is_numerator:
        return nuisance_degrees_of_freedom

    return (
        config.train__nn_significant_degrees_of_freedom
        + nuisance_degrees_of_freedom
    )


def statistic_degrees_of_freedom(
        config: Union[TrainConfig, DetectorConfig],
) -> int:
    """Return the numerator-model degrees of freedom minus the denominator's."""
    return (
        model_degrees_of_freedom(config, is_numerator=True)
        - model_degrees_of_freedom(config, is_numerator=False)
    )
