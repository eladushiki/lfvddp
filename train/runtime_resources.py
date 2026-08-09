"""Observe resources after a scheduled job has started.

This module is not a second cluster scheduler.  The scheduler-facing code in
``frame.command_line.execution`` requests CPUs, GPUs, and memory from PBS.  Once
PBS starts the job, this module observes the process affinity and scheduler-
scoped CUDA visibility that actually reached the container.  Profiling and
artifact generation belong to ``train.training_profiler``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Mapping, Optional

import torch

ALLOCATED_CPUS_ENV = "LFVDDP_ALLOCATED_CPUS"
ALLOCATED_GPU_IDS_ENV = "LFVDDP_ALLOCATED_GPU_IDS"


def _positive_int(value: Optional[str]) -> Optional[int]:
    """Parse an optional environment value as a strictly positive integer."""

    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"Expected a positive integer, got {value!r}.") from error
    if parsed < 1:
        raise ValueError(f"Expected a positive integer, got {value!r}.")
    return parsed


def _affinity_cpu_ids() -> tuple[int, ...]:
    """Return CPUs on which this process may run, or empty if unsupported."""

    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return ()


def _split_gpu_ids(value: Optional[str]) -> tuple[str, ...]:
    """Parse a scheduler-style comma-separated GPU visibility value."""

    if value is None:
        return ()
    stripped = value.strip()
    if not stripped or stripped == "-1":
        return ()
    return tuple(item.strip() for item in stripped.split(",") if item.strip())


@dataclass(frozen=True)
class RuntimeAllocation:
    """Resources exposed to this process by the scheduler and container."""

    cpu_count: int
    cpu_affinity: tuple[int, ...]
    assigned_gpu_ids: tuple[str, ...]
    visible_gpu_count: int
    gpu_names: tuple[str, ...]
    gpu_total_memory_bytes: tuple[int, ...]
    requested_cpus: Optional[int] = None
    requested_gpus: int = 0

    @property
    def usable_gpu_count(self) -> int:
        """Count scheduler-assigned GPUs, never unrestricted host GPUs."""

        return len(self.assigned_gpu_ids)

    def as_dict(self) -> dict:
        """Return a JSON-compatible representation including derived fields."""

        result = asdict(self)
        result["cpu_affinity"] = list(self.cpu_affinity)
        result["assigned_gpu_ids"] = list(self.assigned_gpu_ids)
        result["gpu_names"] = list(self.gpu_names)
        result["gpu_total_memory_bytes"] = list(self.gpu_total_memory_bytes)
        result["usable_gpu_count"] = self.usable_gpu_count
        return result


def detect_runtime_allocation(
    requested_cpus: Optional[int],
    requested_gpus: int,
    environment: Optional[Mapping[str, str]] = None,
) -> RuntimeAllocation:
    """Resolve execution capacity from scheduler-scoped runtime visibility.

    Requested values are retained for diagnostics only. CPU execution uses the
    affinity-aware count exported by the PBS script, while GPU execution uses
    scheduler-assigned identifiers cross-checked with PyTorch visibility.
    """

    environment = os.environ if environment is None else environment
    affinity = _affinity_cpu_ids()
    exported_cpu_count = _positive_int(environment.get(ALLOCATED_CPUS_ENV))
    if exported_cpu_count is not None:
        cpu_count = exported_cpu_count
    elif affinity:
        cpu_count = len(affinity)
    else:
        cpu_count = os.cpu_count() or 1

    if affinity:
        cpu_count = min(cpu_count, len(affinity))
    if cpu_count < 1:
        raise RuntimeError("PBS exposed no usable CPUs to the training process.")

    assigned_gpu_ids = _split_gpu_ids(environment.get(ALLOCATED_GPU_IDS_ENV))
    visible_gpu_count = torch.cuda.device_count()

    # Some PBS installations expose only CUDA_VISIBLE_DEVICES. The execution
    # script normally copies it into LFVDDP_ALLOCATED_GPU_IDS before cleanenv.
    if not assigned_gpu_ids:
        assigned_gpu_ids = _split_gpu_ids(environment.get("CUDA_VISIBLE_DEVICES"))

    if requested_gpus > 0 and not assigned_gpu_ids:
        raise RuntimeError(
            "The job requested GPUs, but PBS/container runtime exposed no CUDA devices "
            "as scheduler assignments. Check CUDA_VISIBLE_DEVICES, PBS_GPUFILE, and "
            "Singularity --nv support."
        )

    if assigned_gpu_ids and visible_gpu_count < len(assigned_gpu_ids):
        raise RuntimeError(
            f"PBS assigned {len(assigned_gpu_ids)} GPU(s), but PyTorch sees only "
            f"{visible_gpu_count}. Check CUDA_VISIBLE_DEVICES and Singularity --nv."
        )

    usable_gpu_count = len(assigned_gpu_ids)
    gpu_names = tuple(
        torch.cuda.get_device_name(index) for index in range(usable_gpu_count)
    )
    gpu_total_memory_bytes = tuple(
        torch.cuda.get_device_properties(index).total_memory
        for index in range(usable_gpu_count)
    )

    return RuntimeAllocation(
        cpu_count=cpu_count,
        cpu_affinity=affinity,
        assigned_gpu_ids=assigned_gpu_ids,
        visible_gpu_count=visible_gpu_count,
        gpu_names=gpu_names,
        gpu_total_memory_bytes=gpu_total_memory_bytes,
        requested_cpus=requested_cpus,
        requested_gpus=requested_gpus,
    )

