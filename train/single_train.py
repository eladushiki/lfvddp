import torch

from data_tools.data_generation import DataBatch, DataGeneration
from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetConfig
from data_tools.detector.detector_effect import DetectorEffect
from data_tools.profile_likelihood import calc_t_LFVDDP
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import RESULTING_T_FILE_NAME
from frame.file_system.training_history import HistoryKeys
from neural_networks.utils import save_training_history_outcome
from train.model_trainer import (
    SequentialTrainLauncher,
    TrainLauncher,
)
from train.tensorboard_clutch import log_t_history_to_tensorboard
from train.train_config import TrainConfig
from train.training_names import (
    SAMPLE_A_NAME,
    SAMPLE_B_NAME,
    training_name,
    training_names_for_sample,
)


@context_controlled_execution
def main(context: ExecutionContext) -> None:

    # type casting safety for the config type
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    if not isinstance(config, DatasetConfig):
        raise TypeError(f"Expected DatasetConfig, got {config.__class__.__name__}")

    # Generate data
    gen = DataGeneration(context)
    batch = gen.get_batch()

    # Simulate detector
    det = DetectorEffect(context)
    detected_batch = det.affect_batch(batch)

    t_result = train_for_t(
        context=context,
        data_batch=detected_batch,
        detector_effect=det,
        name=SAMPLE_A_NAME,
    )

    ## Training log
    context.save_and_document_text(
        f"{t_result}\n", file_path=context.unique_out_dir / RESULTING_T_FILE_NAME
    )


def train_for_t(
    context: ExecutionContext,
    data_batch: DataBatch,
    detector_effect: DetectorEffect,
    name: str,
) -> float:
    """
    Call either a parallel launcher or the sequential training, according to config.
    """
    # Train for each max expression in the t value formula to obtain its value.
    if context.config.train__run_symmetric_in_parallel:
        raise NotImplementedError(
            "Symmetric training in parallel is not yet implemented."
        )
    train_launcher = SequentialTrainLauncher(context, detector_effect)

    training_names = training_names_for_sample(name)

    numerator_train_idx = train_launcher.add_training(
        data_batch=data_batch,
        detector_effect=detector_effect,
        is_numerator=True,
        name=training_names.numerator,
    )
    denominator_train_idx = train_launcher.add_training(
        data_batch=data_batch,
        detector_effect=detector_effect,
        is_numerator=False,
        name=training_names.denominator,
    )

    train_launcher.execute_trainings()

    numerator_training = train_launcher.get_training(numerator_train_idx)
    denominator_training = train_launcher.get_training(denominator_train_idx)

    if numerator_training.history is None or denominator_training.history is None:
        # NPLM does not expose the paired minimization histories used here.
        final_t = float(
            calc_t_LFVDDP(
                numerator=numerator_training.result,
                denominator=denominator_training.result,
            )
        )
    else:
        training_dtype = numerator_training.model._dtype

        numerator = torch.as_tensor(
            numerator_training.history[HistoryKeys.LOSS.value],
            dtype=training_dtype,
        ).numpy()
        denominator = torch.as_tensor(
            denominator_training.history[HistoryKeys.LOSS.value],
            dtype=training_dtype,
        ).numpy()

        t_history = {
            HistoryKeys.EPOCH.value: numerator_training.history[
                HistoryKeys.EPOCH.value
            ],
            HistoryKeys.NUMERATOR.value: numerator,
            HistoryKeys.DENOMINATOR.value: denominator,
            HistoryKeys.T.value: calc_t_LFVDDP(numerator, denominator),
        }
        save_training_history_outcome(
            context=context,
            model_history=t_history,
            model_name=name,
        )
        log_t_history_to_tensorboard(context, name, t_history)
        final_t = float(t_history[HistoryKeys.T.value][-1])

    plot_training_prediction(
        context=context,
        numerator_training=numerator_training,
        denominator_training=denominator_training,
    )
    return final_t


def plot_training_prediction(
    context: ExecutionContext,
    numerator_training: TrainLauncher.Training,
    denominator_training: TrainLauncher.Training,
) -> None:
    if (
        not context.is_debug_mode
        or numerator_training.model is None
        or denominator_training.model is None
    ):
        return

    from plot.plots import plot_prediction_process_sliced

    base_name = numerator_training.data_batch.parameters[
        DataSet.DataSetCategory.A_SR
    ].name
    model_name = numerator_training.name or training_name(base_name, is_numerator=True)

    data_process_plot = plot_prediction_process_sliced(
        context=context,
        numerator_training=numerator_training,
        denominator_training=denominator_training,
        title=base_name + " prediction process",
    )
    context.save_and_document_figure(
        data_process_plot,
        context.unique_out_dir / f"{model_name}_data_process_plot.png",
    )


if __name__ == "__main__":
    main()
