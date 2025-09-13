from os import makedirs

from data_tools.data_generation import DataGeneration
from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetConfig
from frame.file_structure import SINGLE_TRAINING_RESULT_FILE_NAME

from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from neural_networks.NPLM_adapters import calc_t_NPLM
from neural_networks.differentiating_model import calc_t_LFVNN
from plot.plots import plot_prediction_process_sliced
from train.train_config import TrainConfig


@context_controlled_execution
def main(context: ExecutionContext) -> None:

    # type casting safety for the config type
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    if not isinstance(config, DatasetConfig):
        raise TypeError(f"Expected DatasetConfig, got {config.__class__.__name__}")

    gen = DataGeneration(context)

    # Generate data
    A_dataset = gen["TauMuon"]
    B_dataset = gen["TauElectron"]
    reference_dataset = A_dataset + B_dataset

    # Train symmetrically to obtain the combined loss
    t_a_loss = follow_instructions_for_t(context, A_dataset, reference_dataset, name="A_model")
    t_b_loss = follow_instructions_for_t(context, B_dataset, reference_dataset, name="B_model")
    final_t = t_a_loss + t_b_loss

    ## Training log
    makedirs(context.training_outcomes_dir, exist_ok=True)
    context.save_and_document_text(
        f"{final_t}\n",
        file_path=context.training_outcomes_dir / SINGLE_TRAINING_RESULT_FILE_NAME
    )


def follow_instructions_for_t(
        context: ExecutionContext,
        sample_dataset: DataSet,
        reference_dataset: DataSet,
        name: str,
) -> float:
    if not isinstance((config := context.config), TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    if config.train__like_NPLM:
        model, final_t = calc_t_NPLM(
            context,
            sample_dataset,
            reference_dataset,
            name,
        )
    else:
        model, final_t = calc_t_LFVNN(
            context,
            sample_dataset,
            reference_dataset,
            name,
        )

    if context.is_debug_mode:
        data_process_plot = plot_prediction_process_sliced(
            context=context,
            experiment_sample=sample_dataset,
            reference_sample=reference_dataset,
            trained_tau_model=model,
            trained_delta_model=None,
            title=name + " prediction process",
        )
        context.save_and_document_figure(data_process_plot, context.unique_out_dir / f"{name}_data_process_plot.png")

    return final_t


if __name__ == "__main__":
    main()
