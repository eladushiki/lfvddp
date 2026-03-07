from data_tools.detector.detector_effect import DetectorEffect
from data_tools.data_generation import DataGeneration
from data_tools.dataset_config import DatasetConfig
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import RESULTING_T_FILE_NAME
from train.multiprocessing_train import follow_instructions_for_t, symmetric_train_in_parallel
from train.train_config import TrainConfig


@context_controlled_execution
def main(context: ExecutionContext) -> None:

    # type casting safety for the config type
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    if not isinstance(config, DatasetConfig):
        raise TypeError(f"Expected DatasetConfig, got {config.__class__.__name__}")

    dataset_1_name = "TauMuon"
    dataset_2_name = "TauElectron"

    gen = DataGeneration(context)

    # Generate data
    A_dataset, A_params = gen[dataset_1_name]
    B_dataset, B_params = gen[dataset_2_name]

    # Simulate detector
    det = DetectorEffect(context)
    detected_A_dataset = det.affect_and_compensate(A_dataset, A_params, is_display=context.is_debug_mode)
    detected_B_dataset = det.affect_and_compensate(B_dataset, B_params, is_display=context.is_debug_mode)

    # For reference, we combine both datasets
    reference_dataset = detected_A_dataset + detected_B_dataset
    model_a_name = f"A_model_for_{dataset_1_name}"
    model_b_name = f"B_model_for_{dataset_2_name}"

    # Train symmetrically to obtain the combined loss.
    # Mode is configurable: parallel subprocesses or sequential in-process execution.
    if config.train__run_symmetric_in_parallel:
        t_a, t_b = symmetric_train_in_parallel(
            context=context,
            detected_A_dataset=detected_A_dataset,
            detected_B_dataset=detected_B_dataset,
            reference_dataset=reference_dataset,
            detector_effect=det,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
        )
    else:
        _, t_a = follow_instructions_for_t(
            context=context,
            sample_dataset=detected_A_dataset,
            reference_dataset=reference_dataset,
            detector_effect=det,
            name=model_a_name,
        )
        _, t_b = follow_instructions_for_t(
            context=context,
            sample_dataset=detected_B_dataset,
            reference_dataset=reference_dataset,
            detector_effect=det,
            name=model_b_name,
        )

    final_t = t_a + t_b

    ## Training log
    context.save_and_document_text(
        f"{final_t}\n",
        file_path=context.unique_out_dir / RESULTING_T_FILE_NAME
    )


if __name__ == "__main__":
    main()
