from abc import abstractmethod
from dataclasses import dataclass
from logging import info
from multiprocessing import get_context
from pathlib import Path
import traceback
from typing import Any, Optional

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from data_tools.detector.detector_effect import DetectorEffect
from frame.context.execution_context import ExecutionContext
from frame.file_structure import SINGLE_TRAIN_T_FILE_NAME
from frame.file_system.training_history import HistoryKeys
from train.checkpoints import find_latest_training_checkpoint
from train.training_names import training_name
from neural_networks.utils import ContextedModel


class TrainLauncher:
    @dataclass
    class Training:
        data_batch: DataBatch
        detector_effect: DetectorEffect
        is_numerator: bool
        name: Optional[str] = None
        result: Optional[float] = None
        model: Optional[ContextedModel] = None

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
        name: Optional[str] = None,
    ) -> int:
        self._train_stack.append(
            self.Training(
                data_batch=data_batch,
                detector_effect=detector_effect,
                is_numerator=is_numerator,
                name=name,
            )
        )
        return len(self._train_stack) - 1  # Return the index of the added training

    @abstractmethod
    def execute_trainings(self):
        pass

    def get_train_result(self, idx):
        return self._train_stack[idx].result

    def _training_model_name(self, training: Training) -> str:
        if training.name is not None:
            return training.name
        base_name = training.data_batch.parameters[DataSet.DataSetCategory.A_SR].name
        return training_name(base_name, training.is_numerator)

    def _training_checkpoint(self, training: Training) -> Optional[dict[str, Any]]:
        checkpoint_result = find_latest_training_checkpoint(
            self._context,
            self._training_model_name(training),
            warn_missing=False,
        )
        if checkpoint_result is None:
            return None
        _, checkpoint = checkpoint_result
        return checkpoint

    def _checkpoint_finished_training(self, checkpoint: dict[str, Any]) -> bool:
        return int(checkpoint.get("epoch", -1)) >= self._config.train__epochs - 1

    def _checkpoint_result(self, checkpoint: dict[str, Any]) -> float:
        training_history = checkpoint.get("training_history", {})
        losses = training_history.get(HistoryKeys.LOSS.value)
        if losses is None or len(losses) == 0:
            raise RuntimeError(
                "Cannot recover training result from checkpoint without loss history."
            )
        return float(losses[-1])

    def _follow_instructions_for_t(
        self,
        training: Training,
    ) -> tuple[Optional[ContextedModel], float]:

        model_name = self._training_model_name(training)

        if self._config.train__like_NPLM:
            from neural_networks.NPLM_adapters import calc_t_NPLM

            sample_a_dataset = training.data_batch.datasets[
                DataSet.DataSetCategory.A_SR
            ]
            sample_b_dataset = training.data_batch.datasets[
                DataSet.DataSetCategory.B_SR
            ]

            model, final_val = calc_t_NPLM(
                self._context,
                sample_a_dataset,
                sample_b_dataset,
                f"NPLM_train_for_{model_name}",
            )
        else:
            from neural_networks.differentiating_model import (
                DifferentiatingModel,
                calc_min_LFVNN,
            )

            if DifferentiatingModel.has_configured_trainable_parameters(
                self._config,
                training.is_numerator,
            ):
                model, final_val = calc_min_LFVNN(
                    context=self._context,
                    data=training.data_batch,
                    detector_effect=self._detector_effect,
                    is_numerator=training.is_numerator,
                    name=model_name,
                )
            else:
                model = None
                final_val = DifferentiatingModel.calculate_loss_statically(
                    context=self._context,
                    data=training.data_batch,
                    detector_effect=self._detector_effect,
                    is_numerator=training.is_numerator,
                    name=model_name,
                )
                info(f"Calculated static loss for {model_name}: {final_val:.6f}")

        training.model = model
        training.result = final_val
        return model, final_val

    def _plot_training_prediction(self, training: Training) -> None:
        if not self._context.is_debug_mode or training.model is None:
            return
        from plot.plots import plot_prediction_process_sliced

        base_name = training.data_batch.parameters[DataSet.DataSetCategory.A_SR].name
        model_name = self._training_model_name(training)

        data_process_plot = plot_prediction_process_sliced(
            context=self._context,
            detector_effect=self._detector_effect,
            experiment_sample=training.data_batch.datasets[
                DataSet.DataSetCategory.A_SR
            ],
            reference_sample=(
                training.data_batch.datasets[DataSet.DataSetCategory.A_SR]
                + training.data_batch.datasets[DataSet.DataSetCategory.B_SR]
            ),
            trained_tau_model=training.model,
            trained_delta_model=None,
            title=base_name + " prediction process",
            along_observables=self._detector_effect._observable_names[:2],
        )
        self._context.save_and_document_figure(
            data_process_plot,
            self._context.unique_out_dir / f"{model_name}_data_process_plot.png",
        )

    def _plot_training_predictions(self) -> None:
        for training in self._train_stack:
            self._plot_training_prediction(training)


class SequentialTrainLauncher(TrainLauncher):
    def __init__(self, context: ExecutionContext, detector_effect: DetectorEffect):
        super().__init__(context, detector_effect)

    def execute_trainings(self):
        for training in self._train_stack:
            checkpoint = self._training_checkpoint(training)
            if checkpoint is not None and self._checkpoint_finished_training(
                checkpoint
            ):
                training.result = self._checkpoint_result(checkpoint)
                info(
                    f"Skipping completed training {self._training_model_name(training)}."
                )
                continue

            self._follow_instructions_for_t(training)
        self._plot_training_predictions()


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
            error_text = "\n".join(
                f"{name} failed:\n{tb}" for name, tb in errors.items()
            )
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
                file_path=context.unique_out_dir / SINGLE_TRAIN_T_FILE_NAME,
            )

            result_queue.put((name, final_t, None))

        except Exception:
            result_queue.put((name, None, traceback.format_exc()))
