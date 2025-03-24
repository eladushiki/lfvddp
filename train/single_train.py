from os import makedirs
from data_tools.dataset_config import DatasetConfig
from frame.file_structure import SINGLE_TRAINING_RESULT_FILE_NAME
from neural_networks.NPLM_adapters import get_tau_predicting_model, train_model_for_tau

from data_tools.data_utils import DataGeneration, DetectorSimulation, resample
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from train.train_config import TrainConfig


@context_controlled_execution
def main(context: ExecutionContext) -> None:

    # type casting safety for the config type
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    if not isinstance(config, DatasetConfig):
        raise TypeError(f"Expected DatasetConfig, got {config.__class__.__name__}")

    gen = DataGeneration(config)
    det = DetectorSimulation(config)

    # Generate data
    raw_A_dataset = gen.generate_dataset(config.dataset__dataset_A_composition)
    A_dataset = det.simulate_detector_effect(
        raw_A_dataset,
        config.dataset__dataset_A_detector_efficiency,
        config.dataset__dataset_A_detector_error,
        )
    raw_B_dataset = gen.generate_dataset(config.dataset__dataset_B_composition)
    B_dataset = det.simulate_detector_effect(
        raw_B_dataset,
        config.dataset__dataset_B_detector_efficiency,
        config.dataset__dataset_B_detector_error,
    )
    
    if config.dataset__resample_is_resample:  # todo: this was never checked
        raise NotImplementedError("Resampling is not implemented")
        feature_dataset, target_structure = resample(
            feature = feature_dataset,
            target = target_structure,
            background_data_str = config.train__data_background ,
            label_method = config.train__resample_label_method,
            method_type = config.train__resample_method_type,
            replacement = config.train__resample_is_replacement,
        )

    t_a_model = get_tau_predicting_model(config, name="a_model")
    t_b_model = get_tau_predicting_model(config, name="b_model")

    # Train symmetrically to obtain the combined loss
    t_a_loss = train_model_for_tau(context, t_a_model, A_dataset, B_dataset)
    t_b_loss = train_model_for_tau(context, t_b_model, B_dataset, A_dataset)
    final_t = t_a_loss + t_b_loss

    ## Training log
    makedirs(context.training_outcomes_dir, exist_ok=True)
    context.save_and_document_text(
        f"{final_t}\n",
        path=context.training_outcomes_dir / SINGLE_TRAINING_RESULT_FILE_NAME
    )

if __name__ == "__main__":
    main()
