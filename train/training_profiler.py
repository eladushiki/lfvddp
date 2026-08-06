from contextlib import nullcontext
from pathlib import Path
from typing import ContextManager, Optional

import torch

from frame.context.execution_context import ExecutionContext
from train.cpu_runtime import cpu_runtime_metadata


class TrainingProfiler:
    """Own the lifecycle and output artifacts of an opt-in training profile."""

    def __init__(
        self,
        context: ExecutionContext,
        model_name: str,
        number_of_observables: int,
        number_of_events: int,
        number_of_training_epochs: int,
    ) -> None:
        self._context = context
        self._model_name = model_name
        self._number_of_observables = number_of_observables
        self._number_of_events = number_of_events
        self._enabled = context.config.train__profiling_enabled
        self._warmup_epochs = min(
            context.config.train__profiling_warmup_epochs,
            number_of_training_epochs - 1,
        )
        self._active_epochs = min(
            context.config.train__profiling_active_epochs,
            number_of_training_epochs - self._warmup_epochs,
        )
        self._profiler: Optional[torch.profiler.profile] = None
        self._trace_path: Optional[Path] = None
        self._summary_path: Optional[Path] = None

    def __enter__(self) -> "TrainingProfiler":
        if not self._enabled:
            return self

        output_dir = self._context.training_outcomes_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        profile_stem = f"{self._model_name}.{self._number_of_observables}D.profile"
        self._trace_path = output_dir / f"{profile_stem}.trace.json"
        self._summary_path = output_dir / f"{profile_stem}.txt"

        self._profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=self._warmup_epochs,
                active=self._active_epochs,
                repeat=1,
            ),
            on_trace_ready=self._save_profile,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        self._profiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._profiler is not None:
            self._profiler.__exit__(exc_type, exc_value, traceback)

    def region(self, name: str) -> ContextManager:
        if not self._enabled:
            return nullcontext()
        return torch.profiler.record_function(name)

    def step(self) -> None:
        if self._profiler is not None:
            self._profiler.step()

    def _save_profile(self, profiler: torch.profiler.profile) -> None:
        if self._trace_path is None or self._summary_path is None:
            raise RuntimeError("Profile output paths were not initialized.")

        profiler.export_chrome_trace(str(self._trace_path))
        metadata_lines = [
            f"model: {self._model_name}",
            f"observables: {self._number_of_observables}",
            f"events: {self._number_of_events}",
            f"warmup epochs: {self._warmup_epochs}",
            f"profiled epochs: {self._active_epochs}",
        ]
        metadata_lines.extend(
            f"{key}: {value}"
            for key, value in cpu_runtime_metadata(
                self._context.config.cluster__qsub_ncpus
            ).items()
        )
        metadata = "\n".join(metadata_lines) + "\n\n"
        summary = profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total",
            row_limit=200,
        )
        self._summary_path.write_text(metadata + summary)
        self._context.document_created_product(self._trace_path)
        self._context.document_created_product(self._summary_path)
