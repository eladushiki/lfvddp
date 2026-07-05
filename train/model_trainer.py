from abc import abstractmethod
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
import traceback
from typing import Any, Callable, List, Optional

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from data_tools.detector.detector_effect import DetectorEffect
from frame.context.execution_context import ExecutionContext
from frame.file_structure import SINGLE_TRAIN_T_FILE_NAME
from neural_networks.utils import ContextedModel
from plot.plots import plot_prediction_process_sliced


class TrainLauncher:

    @dataclass
    class Training:
        data_batch: DataBatch
        detector_effect: DetectorEffect
        is_numerator: bool
        result: Optional[float] = None

    def __init__(self, context: ExecutionContext, detector_effect: DetectorEffect):
        self._context = context
        self._config = self._context.config
        self._detector_effect = detector_effect
        self._train_stack = []

    def add_training(
        self,
        data_batch: DataBatch,
        detector_effect: DetectorEffect,
        is_numerator: bool,
    ) -> int:
        self._train_stack.append(
            self.Training(
                data_batch=data_batch,
                detector_effect=detector_effect,
                is_numerator=is_numerator,
            )
        )
        return len(self._train_stack) - 1  # Return the index of the added training

    @abstractmethod
    def execute_trainings(self):
        pass

    def get_train_result(self, idx):
        return self._train_stack[idx].result

    def _follow_instructions_for_t(
        self,
        training: Training,
    ) -> tuple[ContextedModel, float]:
        
        name = training.data_batch.parameters[DataSet.DataSetCategory.A_SR].name

        if self._config.train__like_NPLM:
            from neural_networks.NPLM_adapters import calc_t_NPLM
            sample_a_dataset = training.data_batch.datasets[DataSet.DataSetCategory.A_SR]
            sample_b_dataset = training.data_batch.datasets[DataSet.DataSetCategory.B_SR]

            model, final_val = calc_t_NPLM(
                self._context,
                sample_a_dataset,
                sample_b_dataset,
                f"NPLM train for {name}",
            )
        else:
            from neural_networks.differentiating_model import calc_min_LFVNN
            model, final_val = calc_min_LFVNN(
                context=self._context,
                data=training.data_batch,
                detector_effect=self._detector_effect,
                is_numerator=training.is_numerator,
                name=name,
            )

        if self._context.is_debug_mode:
            data_process_plot = plot_prediction_process_sliced(
                context=self._context,
                detector_effect=self._detector_effect,
                experiment_sample=training.data_batch.datasets[DataSet.DataSetCategory.A_SR],
                reference_sample=training.data_batch.datasets[DataSet.DataSetCategory.A_SR] + training.data_batch.datasets[DataSet.DataSetCategory.B_SR],
                trained_tau_model=model,
                trained_delta_model=None,
                title=name + " prediction process",
                along_observables=self._detector_effect._observable_names[:2],
            )
            self._context.save_and_document_figure(data_process_plot, self._context.unique_out_dir / f"{name}_data_process_plot.png")

        training.result = final_val


class SequentialTrainLauncher(TrainLauncher):
    def __init__(self, context: ExecutionContext, detector_effect: DetectorEffect):
        super().__init__(context, detector_effect)

    def execute_trainings(self):
        for training in self._train_stack:
            self._follow_instructions_for_t(training)
    

class ParallelTrainLauncher(TrainLauncher):
    def __init__(self, context: ExecutionContext, detector_effect: DetectorEffect):
        super().__init__(context, detector_effect)

    def execute_trainings(self):
        mp_context = get_context("fork")
        result_queue = mp_context.Queue()
        child_runs_parent_out_dir = self._context.unique_out_dir

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
