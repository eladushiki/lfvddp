from copy import deepcopy
from os import makedirs
from data_tools.data_generation import DataGeneration
from data_tools.dataset_config import DatasetConfig
from frame.file_structure import SINGLE_TRAINING_RESULT_FILE_NAME
from neural_networks.NPLM_adapters import get_tau_predicting_model, train_model_for_tau

from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from plot.plots import plot_prediction_process
from train.train_config import TrainConfig


@context_controlled_execution
def main(context: ExecutionContext) -> None:

    # type casting safety for the config type
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    if not isinstance(config, DatasetConfig):
        raise TypeError(f"Expected DatasetConfig, got {config.__class__.__name__}")

    gen = DataGeneration(config)

    # Generate data
    A_dataset = gen["A"]
    B_dataset = gen["B"]
    reference_dataset = A_dataset + B_dataset

    A_parameters = config.get_parameters("A")
    B_parameters = config.get_parameters("B")

    t_a_model = get_tau_predicting_model(config, A_parameters, name="a_model")
    t_b_model = get_tau_predicting_model(config, B_parameters, name="b_model")

    # Train symmetrically to obtain the combined loss
    t_a_loss = train_model_for_tau(context, t_a_model, A_dataset, reference_dataset)
    t_b_loss = train_model_for_tau(context, t_b_model, B_dataset, reference_dataset)
    final_t = t_a_loss + t_b_loss

    ## Training log
    makedirs(context.training_outcomes_dir, exist_ok=True)
    context.save_and_document_text(
        f"{final_t}\n",
        path=context.training_outcomes_dir / SINGLE_TRAINING_RESULT_FILE_NAME
    )

    if context.is_debug_mode:
        data_process_plot = plot_prediction_process(
            context=context,
            experiment_sample=A_dataset,
            trained_model=t_a_model,
            reference_sample=reference_dataset,
        )
        context.save_and_document_figure(data_process_plot, context.unique_out_dir / "data_process_plot.png")

if __name__ == "__main__":
    main()
