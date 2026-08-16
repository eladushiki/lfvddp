"""Coarse run telemetry and opt-in detailed PyTorch training profiles.

``TrainingResourceProfiler`` owns the always-on, run-level resource artifact.
``TrainingProfiler`` owns the optional operator-level Torch trace for one model.
Keeping both here gives profiling one home while preserving their distinct
lifecycles and output formats.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import os
from pathlib import Path
import resource
from time import perf_counter
from typing import ContextManager, Iterator, Optional

import torch

from frame.context.execution_context import ExecutionContext
from train.cpu_runtime import cpu_runtime_metadata
from train.runtime_resources import (
    ALLOCATED_CPUS_ENV,
    ALLOCATED_GPU_IDS_ENV,
    RuntimeAllocation,
)


RUNTIME_RESOURCES_FILE_NAME = "runtime_resources.json"


@dataclass
class TrainingResourceProfiler:
    """Profile run stages, branch placement, throughput, and peak resources."""

    context: ExecutionContext
    allocation: RuntimeAllocation
    requested_memory_gib: Optional[int] = None
    started_at: float = field(default_factory=perf_counter)
    stage_seconds: dict[str, float] = field(default_factory=dict)
    branch_assignments: dict[str, dict] = field(default_factory=dict)
    unused_resources: list[str] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Measure a named run stage, including time spent raising errors."""

        started_at = perf_counter()
        try:
            yield
        finally:
            self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + (
                perf_counter() - started_at
            )

    def record_branch(
        self,
        name: str,
        *,
        device: str,
        cpu_threads: int,
        elapsed_seconds: float,
        epochs: int,
        static: bool,
        peak_rss: Optional[int] = None,
        peak_cuda_allocated_bytes: Optional[int] = None,
        peak_cuda_reserved_bytes: Optional[int] = None,
    ) -> None:
        """Record one branch's assignment, throughput, and process/device peaks."""

        self.branch_assignments[name] = {
            "device": device,
            "CPU threads": cpu_threads,
            "elapsed seconds": elapsed_seconds,
            "epochs": epochs,
            "epochs per second": (
                epochs / elapsed_seconds if elapsed_seconds > 0 and not static else None
            ),
            "static": static,
            "peak resident set size": peak_rss,
            "peak CUDA allocated bytes": peak_cuda_allocated_bytes,
            "peak CUDA reserved bytes": peak_cuda_reserved_bytes,
        }

    def save(self) -> Path:
        """Write requested-versus-observed resources and accumulated telemetry."""

        usage = resource.getrusage(resource.RUSAGE_SELF)
        cuda_peaks = {}
        for index in range(self.allocation.usable_gpu_count):
            cuda_peaks[str(index)] = {
                "allocated bytes": torch.cuda.max_memory_allocated(index),
                "reserved bytes": torch.cuda.max_memory_reserved(index),
            }
        contents = {
            "requested": {
                "CPUs": self.allocation.requested_cpus,
                "GPUs": self.allocation.requested_gpus,
                "memory GiB": self.requested_memory_gib,
            },
            "observed": self.allocation.as_dict(),
            "assignments": self.branch_assignments,
            "unused resources": self.unused_resources,
            "timing seconds": {
                **self.stage_seconds,
                "total": perf_counter() - self.started_at,
            },
            "peaks": {
                "maximum resident set size": usage.ru_maxrss,
                "maximum resident set size units": (
                    "bytes" if os.uname().sysname == "Darwin" else "KiB"
                ),
                "CUDA": cuda_peaks,
            },
            "environment": {
                key: os.environ.get(key, "unset")
                for key in (
                    ALLOCATED_CPUS_ENV,
                    ALLOCATED_GPU_IDS_ENV,
                    "CUDA_VISIBLE_DEVICES",
                    "PBS_JOBID",
                    "PBS_ARRAY_INDEX",
                    "PBS_NCPUS",
                    "PBS_GPUFILE",
                )
            },
        }
        return self.context.save_and_document_dict(
            contents,
            self.context.unique_out_dir / RUNTIME_RESOURCES_FILE_NAME,
        )


class TrainingProfiler:
    """Own the lifecycle and output artifacts of an opt-in training profile."""

    def __init__(
        self,
        context: ExecutionContext,
        model_name: str,
        number_of_observables: int,
        number_of_events: int,
        number_of_training_epochs: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self._context = context
        self._model_name = model_name
        self._number_of_observables = number_of_observables
        self._number_of_events = number_of_events
        self._device = device
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

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self._device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self._profiler = torch.profiler.profile(
            activities=activities,
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
            for key, value in cpu_runtime_metadata().items()
        )
        metadata_lines.append(f"training device: {self._device}")
        metadata = "\n".join(metadata_lines) + "\n\n"
        summary = profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cpu_time_total",
            row_limit=200,
        )
        self._summary_path.write_text(metadata + summary)
        self._context.document_created_product(self._trace_path)
        self._context.document_created_product(self._summary_path)
