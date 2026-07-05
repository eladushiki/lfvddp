from data_tools.detector.detector_effect import DetectorEffect
from data_tools.data_generation import DataBatch, DataGeneration
from data_tools.dataset_config import DatasetConfig
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import RESULTING_T_FILE_NAME
from train.model_trainer import ParallelTrainLauncher, SequentialTrainLauncher
from train.train_config import TrainConfig
from train.training_names import (
    SAMPLE_A_NAME,
    SAMPLE_B_NAME,
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
    detected_batch = det.affect_and_compensate_batch(batch)

    t_a = train_for_t(
        context=context,
        data_batch=detected_batch,
        detector_effect=det,
        name=SAMPLE_A_NAME
    )

    detected_batch.swap_ab()

    t_b = train_for_t(
        context=context,
        data_batch=detected_batch,
        detector_effect=det,
        name=SAMPLE_B_NAME
    )

    t_result = t_a + t_b

    ## Training log
    context.save_and_document_text(
        f"{t_result}\n",
        file_path=context.unique_out_dir / RESULTING_T_FILE_NAME
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
        raise NotImplementedError("Symmetric training in parallel is not yet implemented.")
        train_launcer = ParallelTrainLauncher(context, detector_effect)
    else:
        train_launcer = SequentialTrainLauncher(context, detector_effect)

    training_names = training_names_for_sample(name)
    
    numerator_train_idx = train_launcer.add_training(
        data_batch=data_batch,
        detector_effect=detector_effect,
        is_numerator=True,
        name=training_names.numerator,
    )
    denominator_train_idx = train_launcer.add_training(
        data_batch=data_batch,
        detector_effect=detector_effect,
        is_numerator=False,
        name=training_names.denominator,
    )

    train_launcer.execute_trainings()

    numerator_result = train_launcer.get_train_result(numerator_train_idx)
    denominator_result = train_launcer.get_train_result(denominator_train_idx)

    return -2 * numerator_result + 2 * denominator_result


if __name__ == "__main__":
    main()
