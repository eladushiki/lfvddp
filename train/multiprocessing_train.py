from multiprocessing import get_context
from pathlib import Path
import traceback

from data_tools.data_utils import DataSet
from data_tools.detector.detector_effect import DetectorEffect
from frame.context.execution_context import ExecutionContext
from frame.file_structure import SINGLE_TRAIN_T_FILE_NAME
from neural_networks.utils import ContextedModel
from plot.plots import plot_prediction_process_sliced
from train.train_config import TrainConfig


def follow_instructions_for_t(
        context: ExecutionContext,
        sample_dataset: DataSet,
        reference_dataset: DataSet,
        detector_effect: DetectorEffect,
        name: str,
) -> tuple[ContextedModel, float]:
    if not isinstance((config := context.config), TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    if config.train__like_NPLM:
        from neural_networks.NPLM_adapters import calc_t_NPLM
        model, final_t = calc_t_NPLM(
            context,
            sample_dataset,
            reference_dataset,
            name,
        )
    else:
        from neural_networks.differentiating_model import calc_t_LFVNN
        model, final_t = calc_t_LFVNN(
            context,
            sample_dataset,
            reference_dataset,
            detector_effect,
            name,
        )

    if context.is_debug_mode:
        data_process_plot = plot_prediction_process_sliced(
            context=context,
            detector_effect=detector_effect,
            experiment_sample=sample_dataset,
            reference_sample=reference_dataset,
            trained_tau_model=model,
            trained_delta_model=None,
            title=name + " prediction process",
            along_observables=detector_effect._observable_names[:2],
        )
        context.save_and_document_figure(data_process_plot, context.unique_out_dir / f"{name}_data_process_plot.png")

    return model, final_t


def _parallel_training_worker(
    result_queue,
    context: ExecutionContext,
    sample_dataset: DataSet,
    reference_dataset: DataSet,
    detector_effect: DetectorEffect,
    name: str,
    child_runs_parent_out_dir: Path,
) -> None:
    try:
        # Context differences between child processes
        context.config.config__out_dir = child_runs_parent_out_dir
        context.config.config__runtag = name

        _, final_t = follow_instructions_for_t(
            context=context,
            sample_dataset=sample_dataset,
            reference_dataset=reference_dataset,
            detector_effect=detector_effect,
            name=name,
        )

        context.save_and_document_text(
            f"{final_t}\n",
            file_path=context.unique_out_dir / SINGLE_TRAIN_T_FILE_NAME
        )

        result_queue.put((name, final_t, None))

    except Exception:
        result_queue.put((name, None, traceback.format_exc()))


def symmetric_train_in_parallel(
    context: ExecutionContext,
    detected_A_dataset: DataSet,
    detected_B_dataset: DataSet,
    reference_dataset: DataSet,
    detector_effect: DetectorEffect,
    model_a_name: str,
    model_b_name: str,
) -> tuple[float, float]:
    mp_context = get_context("fork")
    result_queue = mp_context.Queue()
    child_runs_parent_out_dir = context.unique_out_dir

    processes = [
        mp_context.Process(
            target=_parallel_training_worker,
            args=(
                result_queue,
                context,
                detected_A_dataset,
                reference_dataset,
                detector_effect,
                model_a_name,
                child_runs_parent_out_dir,
            ),
        ),
        mp_context.Process(
            target=_parallel_training_worker,
            args=(
                result_queue,
                context,
                detected_B_dataset,
                reference_dataset,
                detector_effect,
                model_b_name,
                child_runs_parent_out_dir,
            ),
        ),
    ]

    for process in processes:
        process.start()

    results = {}
    errors = {}
    for _ in processes:
        model_name, final_t, maybe_traceback = result_queue.get()
        if maybe_traceback is not None:
            errors[model_name] = maybe_traceback
        else:
            results[model_name] = final_t

    for process in processes:
        process.join()

    if errors:
        error_text = "\n".join(f"{name} failed:\n{tb}" for name, tb in errors.items())
        raise RuntimeError(f"Parallel symmetric training failed.\n{error_text}")

    if model_a_name not in results or model_b_name not in results:
        raise RuntimeError(
            f"Parallel symmetric training did not return both {model_a_name} and {model_b_name} results."
        )

    return float(results[model_a_name]), float(results[model_b_name])
