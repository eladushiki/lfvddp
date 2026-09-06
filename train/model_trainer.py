"""Execute queued training branches using a caller-selected strategy.

The caller, not a launcher, decides which concrete subclass represents the run:

``TrainLauncher``
    Abstract queue, naming, checkpoint, and model-dispatch interface.
``_ResourceAwareTrainLauncher``
    Abstract implementation layer shared by the two LFVNN strategies.  It owns
    resource assignments and result transport, but never selects a strategy.
``SequentialTrainLauncher``
    Executes every pending branch in the parent process.
``ParallelTrainLauncher``
    Executes trainable branches in spawn-safe worker processes.

``train.single_train.select_train_launcher_class`` makes the run-level decision
before the concrete launcher is constructed.  This preserves the original
purpose of the inheritance hierarchy: each subclass has a fixed execution
contract rather than dynamically changing strategies.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import logging
from logging import info
from multiprocessing import get_context
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
import traceback
from typing import Any, Iterator, Optional

import torch
from werkzeug.utils import secure_filename

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from data_tools.detector.detector_effect import DetectorEffect
from frame.context.execution_context import ExecutionContext
from frame.file_structure import CONTEXT_FILE_NAME
from frame.file_system.training_history import HistoryKeys
from neural_networks.differentiating_model import DifferentiatingModel, calc_min_LFVNN
from neural_networks.utils import ContextedModel
from train.checkpoints import find_latest_training_checkpoint
from train.cpu_runtime import configure_cpu_runtime, cpu_thread_environment
from train.runtime_resources import RuntimeAllocation
from train.train_config import TrainConfig
from train.training_profiler import TrainingResourceProfiler
from train.training_names import training_name


class TrainLauncher(ABC):
    """Common queue, naming, checkpoint, and model-execution operations.

    This base class intentionally contains no process-selection policy.
    """

    @dataclass
    class Training:
        data_batch: DataBatch
        detector_effect: DetectorEffect
        is_numerator: bool
        name: Optional[str] = None
        result: Optional[float] = None
        model: Optional[ContextedModel] = None
        history: Optional[dict[str, Any]] = None

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
        """Queue one branch and return its stable index for later result lookup."""

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
    def execute_trainings(self) -> None:
        """Execute every queued training according to a concrete policy."""
        raise NotImplementedError

    def get_training(self, idx: int) -> Training:
        return self._train_stack[idx]

    def get_train_result(self, idx: int) -> Optional[float]:
        return self.get_training(idx).result

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
        device: str = "cpu",
    ) -> tuple[Optional[ContextedModel], float]:
        """Run the configured model implementation for one prepared record.

        Placement and process policy stay in the concrete launcher.  This common
        operation only dispatches to NPLM or LFVNN and stores their returned
        model, scalar objective, and (when available) history on the record.
        """

        model_name = self._training_model_name(training)

        if self._config.train__like_NPLM:
            from neural_networks.NPLM_adapters import calc_t_NPLM

            sample_a_dataset = training.data_batch.datasets[
                DataSet.DataSetCategory.A_SR
            ]
            sample_b_dataset = training.data_batch.datasets[
                DataSet.DataSetCategory.B_SR
            ]

            model, final_val = calc_t_NPLM( # TODO: no chance this works
                self._context,
                sample_a_dataset,
                sample_b_dataset,
                f"NPLM_train_for_{model_name}",
            )
        else:
            model, final_val, history = calc_min_LFVNN(
                context=self._context,
                data=training.data_batch,
                detector_effect=self._detector_effect,
                is_numerator=training.is_numerator,
                name=model_name,
                device=device,
            )
            training.history = history

        training.model = model
        training.result = final_val
        return model, final_val


@dataclass(frozen=True)
class _TrainingAssignment:
    """One queued branch's device and CPU-thread budget."""

    index: int
    device: str
    cpu_threads: int


def lfvnn_denominator_is_trainable(config: TrainConfig) -> bool:
    """Return whether LFVNN must optimize, rather than calculate, its denominator."""

    return config.train__data_is_train_for_nuisances


def allocation_supports_parallel_training(allocation: RuntimeAllocation) -> bool:
    """Return whether two branches can receive independent CPU or GPU capacity."""

    return allocation.cpu_count >= 2 or allocation.usable_gpu_count >= 2


def _training_seed(context: ExecutionContext, model_name: str) -> int:
    """Derive a stable branch seed independent of execution order or process."""

    digest = hashlib.blake2b(
        f"{context.random_seed}:{model_name}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def _state_dict_on_cpu(model: DifferentiatingModel) -> dict[str, Any]:
    """Convert model state to ordinary CPU arrays for safe pipe transport.

    Sending live Torch tensors can make multiprocessing depend on temporary
    shared-memory files that disappear when the worker exits.  NumPy copies
    keep ownership explicit until the parent reconstructs the model.
    """

    return {
        key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }


@contextmanager
def _capture_worker_output(output_path: Path) -> Iterator[None]:
    """Redirect one worker's combined stdout/stderr to its own text file.

    File-descriptor redirection captures Python, logging, tqdm, and native-library
    output.  Separate files prevent concurrently running workers from writing
    interleaved bytes to the single PBS output stream.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with output_path.open("w", buffering=1) as output_file:
            os.dup2(output_file.fileno(), 1)
            os.dup2(output_file.fileno(), 2)
            yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _parallel_training_worker(
    connection,
    context_path: Path,
    output_path: Path,
    data_batch: DataBatch,
    is_numerator: bool,
    model_name: str,
    device: str,
    cpu_threads: int,
    seed: int,
) -> None:
    """Train one branch in a fresh process and return serializable state.

    A spawned process cannot safely reuse the parent's Torch runtime or mutable
    execution context.  It therefore applies its own thread limit, reloads the
    saved context, reconstructs detector state, and sends results or a complete
    traceback through the one-way connection.
    """
    payload: dict[str, Any]
    try:
        with _capture_worker_output(output_path):
            try:
                configure_cpu_runtime(cpu_threads, log_metadata=False)
                context = ExecutionContext.naive_load_from_file(context_path)
                logging.basicConfig(
                    level=getattr(logging, context.config.config__log_level),
                    force=True,
                )
                info(
                    "Training worker started: %s on %s with %s CPU thread(s).",
                    model_name,
                    device,
                    cpu_threads,
                )
                torch.manual_seed(seed)
                detector_effect = DetectorEffect(context)
                existing_products = len(context.products.products)
                started_at = perf_counter()
                model, final_value, history = calc_min_LFVNN(
                    context=context,
                    data=data_batch,
                    detector_effect=detector_effect,
                    is_numerator=is_numerator,
                    name=model_name,
                    device=device,
                )
                elapsed_seconds = perf_counter() - started_at
                cuda_index = (
                    torch.device(device).index if device.startswith("cuda") else None
                )
                info(
                    "Training worker finished: %s in %.3f seconds.",
                    model_name,
                    elapsed_seconds,
                )
                payload = {
                    "error": None,
                    "result": final_value,
                    "history": history,
                    "state_dict": _state_dict_on_cpu(model),
                    "norm_factor": model._norm_factor,
                    "epochs_executed": model._epochs_executed,
                    "elapsed_seconds": elapsed_seconds,
                    "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "peak_cuda_allocated_bytes": (
                        torch.cuda.max_memory_allocated(cuda_index)
                        if cuda_index is not None
                        else None
                    ),
                    "peak_cuda_reserved_bytes": (
                        torch.cuda.max_memory_reserved(cuda_index)
                        if cuda_index is not None
                        else None
                    ),
                    "products": [
                        str(product.descriptor)
                        for product in context.products.products[existing_products:]
                    ],
                }
            except Exception:
                error_traceback = traceback.format_exc()
                print(error_traceback, file=sys.stderr, flush=True)
                payload = {"error": error_traceback}
    except Exception:
        payload = {"error": traceback.format_exc()}

    try:
        connection.send(payload)
    finally:
        connection.close()


class _ResourceAwareTrainLauncher(TrainLauncher):
    """Share resource-aware mechanics without choosing an execution strategy."""

    def __init__(
        self,
        context: ExecutionContext,
        detector_effect: DetectorEffect,
        allocation: RuntimeAllocation,
        profiler: Optional[TrainingResourceProfiler] = None,
    ) -> None:
        super().__init__(context, detector_effect)
        self._allocation = allocation
        self._profiler = profiler

    def _requires_optimization(self, training: TrainLauncher.Training) -> bool:
        """Return whether a branch must run an optimizer rather than a formula."""

        if self._config.train__like_NPLM:
            return True
        return training.is_numerator or lfvnn_denominator_is_trainable(self._config)

    def _record_training(
        self,
        training: TrainLauncher.Training,
        assignment: _TrainingAssignment,
        elapsed_seconds: float,
        static: bool,
        peak_rss: Optional[int] = None,
        peak_cuda_allocated_bytes: Optional[int] = None,
        peak_cuda_reserved_bytes: Optional[int] = None,
    ) -> None:
        """Add one completed assignment's timing and peaks to run profiling."""

        if self._profiler is None:
            return
        epochs_executed = (
            training.model._epochs_executed
            if isinstance(training.model, DifferentiatingModel)
            else 0
        )
        self._profiler.record_branch(
            self._training_model_name(training),
            device=assignment.device,
            cpu_threads=assignment.cpu_threads,
            elapsed_seconds=elapsed_seconds,
            epochs=epochs_executed,
            static=static,
            peak_rss=peak_rss,
            peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
            peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
        )

    def _completed_checkpoint(self, training: TrainLauncher.Training) -> bool:
        """Restore a finished checkpoint's result and report whether work is done."""

        checkpoint = self._training_checkpoint(training)
        if checkpoint is None or not self._checkpoint_finished_training(checkpoint):
            return False
        training.result = self._checkpoint_result(checkpoint)
        training.history = checkpoint["training_history"]
        info("Skipping completed training %s.", self._training_model_name(training))
        return True

    def _parallel_assignments(
        self, indices: list[int]
    ) -> list[_TrainingAssignment]:
        """Split observed resources between concurrently trainable branches.

        There are at most two independently trainable objectives.  The heavier
        numerator receives the larger CPU share; the denominator receives one
        CPU when it can run alongside it.
        """

        cpu_count = self._allocation.cpu_count
        gpu_count = self._allocation.usable_gpu_count
        if len(indices) == 1:
            self._note_unused_gpus(used_gpu_count=min(1, gpu_count))
            return [
                _TrainingAssignment(
                    indices[0], "cuda:0" if gpu_count else "cpu", cpu_count
                )
            ]

        numerator_index = next(
            index for index in indices if self._train_stack[index].is_numerator
        )
        denominator_index = next(index for index in indices if index != numerator_index)
        numerator_threads = max(1, cpu_count - 1)

        if not allocation_supports_parallel_training(self._allocation):
            raise RuntimeError(
                "ParallelTrainLauncher requires independent capacity for two "
                "branches: at least two CPUs or at least two GPUs."
            )

        if gpu_count >= 2:
            assignments = [
                _TrainingAssignment(numerator_index, "cuda:0", numerator_threads),
                _TrainingAssignment(denominator_index, "cuda:1", 1),
            ]
        elif gpu_count == 1 and cpu_count >= 2:
            assignments = [
                _TrainingAssignment(numerator_index, "cuda:0", numerator_threads),
                _TrainingAssignment(denominator_index, "cpu", 1),
            ]
        else:
            assignments = [
                _TrainingAssignment(numerator_index, "cpu", numerator_threads),
                _TrainingAssignment(denominator_index, "cpu", 1),
            ]

        self._note_unused_gpus(used_gpu_count=min(2, gpu_count))
        return assignments

    def _sequential_assignments(
        self, indices: list[int]
    ) -> list[_TrainingAssignment]:
        """Give each sequential branch the complete reusable allocation."""

        gpu_count = self._allocation.usable_gpu_count
        used_gpu_count = min(1, gpu_count) if indices else 0
        self._note_unused_gpus(used_gpu_count=used_gpu_count)
        device = "cuda:0" if gpu_count else "cpu"
        return [
            _TrainingAssignment(index, device, self._allocation.cpu_count)
            for index in indices
        ]

    def _note_unused_gpus(self, used_gpu_count: int) -> None:
        """Explain why GPUs beyond the independent branch count stay idle."""

        unused_gpu_count = self._allocation.usable_gpu_count - used_gpu_count
        if self._profiler is not None and unused_gpu_count > 0:
            self._profiler.unused_resources.append(
                f"{unused_gpu_count} GPU(s) intentionally unused: there are only "
                f"{used_gpu_count} independently trainable branch device(s)."
            )

    def _execute_in_parent(self, assignment: _TrainingAssignment) -> None:
        """Execute one assignment sequentially in the caller process.

        This is the sequential training path.  It applies the assignment's CPU
        thread budget before any model work, so it has the same CPU-runtime
        configuration as a child worker.
        """

        training = self._train_stack[assignment.index]
        model_name = self._training_model_name(training)
        configure_cpu_runtime(assignment.cpu_threads, log_metadata=False)
        assigned_device = torch.device(assignment.device)
        if assigned_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(assigned_device)
        started_at = perf_counter()
        fork_devices = (
            [assigned_device.index if assigned_device.index is not None else 0]
            if assigned_device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(_training_seed(self._context, model_name))
            self._follow_instructions_for_t(training, device=assignment.device)
        elapsed_seconds = perf_counter() - started_at
        self._record_training(
            training,
            assignment,
            elapsed_seconds,
            static=not self._requires_optimization(training),
            peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            peak_cuda_allocated_bytes=(
                torch.cuda.max_memory_allocated(assigned_device)
                if assigned_device.type == "cuda"
                else None
            ),
            peak_cuda_reserved_bytes=(
                torch.cuda.max_memory_reserved(assigned_device)
                if assigned_device.type == "cuda"
                else None
            ),
        )

    def _restore_worker_result(
        self,
        assignment: _TrainingAssignment,
        payload: dict,
    ) -> None:
        """Rebuild a parent-owned model and training record from worker output."""

        training = self._train_stack[assignment.index]
        model = DifferentiatingModel(
            context=self._context,
            detector_effect=self._detector_effect,
            is_numerator=training.is_numerator,
            name=self._training_model_name(training),
            device=assignment.device,
        )
        model.load_state_dict(
            {
                key: torch.as_tensor(value)
                for key, value in payload["state_dict"].items()
            }
        )
        model._norm_factor = payload["norm_factor"]
        model._epochs_executed = payload["epochs_executed"]
        training.model = model
        training.result = float(payload["result"])
        training.history = payload["history"]
        for product in payload["products"]:
            self._context.document_created_product(Path(product))
        self._record_training(
            training,
            assignment,
            payload["elapsed_seconds"],
            static=False,
            peak_rss=payload["peak_rss"],
            peak_cuda_allocated_bytes=payload["peak_cuda_allocated_bytes"],
            peak_cuda_reserved_bytes=payload["peak_cuda_reserved_bytes"],
        )

    def _worker_output_path(self, model_name: str) -> Path:
        """Return a stable per-training combined stdout/stderr path."""

        safe_model_name = secure_filename(model_name) or "training"
        return (
            self._context.training_outcomes_dir
            / f"{safe_model_name}.worker_output.txt"
        )

    @staticmethod
    def _emit_available_worker_output(
        output_path: Path,
        position: int,
    ) -> int:
        """Replay newly written worker output and return its new file position."""

        if not output_path.exists():
            return position
        with output_path.open("r", errors="replace") as output_file:
            output_file.seek(position)
            output = output_file.read()
            new_position = output_file.tell()
        if not output:
            return new_position
        print(output, end="", flush=True)
        return new_position

    def _collect_worker(
        self,
        assignment: _TrainingAssignment,
        process,
        parent_connection,
        output_path: Path,
    ) -> dict:
        """Stream one worker block to PBS stdout while collecting its result.

        Workers run concurrently, but the parent calls this method in assignment
        order.  Later workers continue writing their private files until their
        turn, preventing their output from interleaving with the current block.
        """

        model_name = self._training_model_name(
            self._train_stack[assignment.index]
        )
        print(
            f"\n===== BEGIN TRAINING OUTPUT: {model_name} "
            f"(device={assignment.device}, CPU threads={assignment.cpu_threads}) =====",
            flush=True,
        )
        position = 0
        payload = None
        try:
            while process.is_alive() or payload is None:
                position = self._emit_available_worker_output(
                    output_path,
                    position,
                )
                if payload is None:
                    if parent_connection.poll(0.1):
                        try:
                            payload = parent_connection.recv()
                        except EOFError:
                            payload = {
                                "error": "Worker exited without returning a result."
                            }
                    elif not process.is_alive():
                        payload = {
                            "error": "Worker exited without returning a result."
                        }
                else:
                    process.join(timeout=0.1)

            process.join()
            position = self._emit_available_worker_output(
                output_path,
                position,
            )
            print(flush=True)
        finally:
            parent_connection.close()

        status = "FAILED" if payload.get("error") is not None else "COMPLETED"
        print(
            f"===== END TRAINING OUTPUT: {model_name} ({status}) =====\n",
            flush=True,
        )
        if output_path.exists():
            self._context.document_created_product(output_path)
        return payload

    def _execute_concurrently(self, assignments: list[_TrainingAssignment]) -> None:
        """Spawn independent assignments and collect results or tracebacks.

        The parent remains responsible for the final model objects, product
        catalogue, reporting, and downstream plots.  Children perform only the
        optimization and return the minimum state needed to reconstruct it.
        """

        mp_context = get_context("spawn")
        context_path = self._context.unique_out_dir / CONTEXT_FILE_NAME
        workers = []
        for assignment in assignments:
            training = self._train_stack[assignment.index]
            parent_connection, child_connection = mp_context.Pipe(duplex=False)
            model_name = self._training_model_name(training)
            output_path = self._worker_output_path(model_name)
            process = mp_context.Process(
                target=_parallel_training_worker,
                args=(
                    child_connection,
                    context_path,
                    output_path,
                    training.data_batch,
                    training.is_numerator,
                    model_name,
                    assignment.device,
                    assignment.cpu_threads,
                    _training_seed(self._context, model_name),
                ),
            )
            workers.append(
                (
                    assignment,
                    process,
                    parent_connection,
                    child_connection,
                    output_path,
                )
            )

        for assignment, process, _, child_connection, _ in workers:
            # The spawn interpreter imports Torch and its native runtimes before
            # entering _parallel_training_worker. Give each child its share of
            # the allocation at process creation so those pools never inherit
            # the full job-wide cap independently.
            with cpu_thread_environment(assignment.cpu_threads):
                process.start()
            child_connection.close()

        errors = []
        for assignment, process, parent_connection, _, output_path in workers:
            payload = self._collect_worker(
                assignment,
                process,
                parent_connection,
                output_path,
            )
            if payload.get("error") is not None:
                errors.append(
                    f"{self._training_model_name(self._train_stack[assignment.index])} "
                    f"failed:\n{payload['error']}"
                )
            else:
                self._restore_worker_result(assignment, payload)

        if errors:
            raise RuntimeError("Parallel LFVNN training failed.\n" + "\n".join(errors))

    def _pending_lfvnn_work(self) -> tuple[list[int], list[int]]:
        """Restore checkpoints and partition pending LFVNN work by trainability."""

        pending_indices = [
            index
            for index, training in enumerate(self._train_stack)
            if not self._completed_checkpoint(training)
        ]
        static_indices = [
            index
            for index in pending_indices
            if not self._requires_optimization(self._train_stack[index])
        ]
        trainable_indices = [
            index for index in pending_indices if index not in static_indices
        ]
        return static_indices, trainable_indices


class SequentialTrainLauncher(_ResourceAwareTrainLauncher):
    """Execute all queued work in the parent process, one branch at a time."""

    def execute_trainings(self) -> None:
        """Run the sequential strategy selected by the caller."""

        if self._config.train__like_NPLM:
            configure_cpu_runtime(self._allocation.cpu_count, log_metadata=False)
            for training in self._train_stack:
                if not self._completed_checkpoint(training):
                    self._follow_instructions_for_t(training)
            return

        static_indices, trainable_indices = self._pending_lfvnn_work()

        for index in static_indices:
            self._execute_in_parent(_TrainingAssignment(index, "cpu", 1))

        for assignment in self._sequential_assignments(trainable_indices):
            self._execute_in_parent(assignment)


class ParallelTrainLauncher(_ResourceAwareTrainLauncher):
    """Execute trainable LFVNN branches in independent spawned workers."""

    def execute_trainings(self) -> None:
        """Run the parallel strategy selected by the caller."""

        if self._config.train__like_NPLM:
            raise RuntimeError("ParallelTrainLauncher does not support NPLM training.")

        static_indices, trainable_indices = self._pending_lfvnn_work()
        for index in static_indices:
            self._execute_in_parent(_TrainingAssignment(index, "cpu", 1))

        assignments = (
            self._parallel_assignments(trainable_indices)
            if trainable_indices
            else []
        )
        if assignments:
            self._execute_concurrently(assignments)
