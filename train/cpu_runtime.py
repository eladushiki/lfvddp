import os
import socket
from logging import info, warning
from pathlib import Path
from typing import Optional

import torch


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
)

# PyTorch documents the inter-op pool size as a one-shot process setting. Some
# releases raise a catchable RuntimeError when it is repeated, while older LCG
# builds terminate in C++ before Python can catch anything. Spawned workers load
# this module afresh, so each process still configures its own pool exactly once.
_INTEROP_THREADS_CONFIGURED = False


def _read_first_existing(paths: tuple[Path, ...]) -> Optional[str]:
    for path in paths:
        try:
            return path.read_text().strip()
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except (FileNotFoundError, PermissionError, OSError, IndexError):
        pass
    return "unknown"


def cpu_runtime_metadata(effective_cpus: Optional[int] = None) -> dict[str, str]:
    """Return stable CPU allocation and PyTorch runtime diagnostics."""
    try:
        affinity = ",".join(str(cpu) for cpu in sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        affinity = "unknown"

    metadata = {
        "hostname": socket.gethostname(),
        "effective CPUs": str(
            effective_cpus
            if effective_cpus is not None
            else torch.get_num_threads()
        ),
        "CPU affinity": affinity,
        "CPU model": _cpu_model(),
        "cgroup cpuset": _read_first_existing(
            (
                Path("/sys/fs/cgroup/cpuset.cpus.effective"),
                Path("/sys/fs/cgroup/cpuset/cpuset.cpus"),
            )
        )
        or "unknown",
        "cgroup CPU quota": _read_first_existing(
            (
                Path("/sys/fs/cgroup/cpu.max"),
                Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
            )
        )
        or "unknown",
        "PyTorch intra-op threads": str(torch.get_num_threads()),
        "PyTorch inter-op threads": str(torch.get_num_interop_threads()),
        "PyTorch version": torch.__version__,
    }
    metadata.update(
        {
            environment_name: os.environ.get(environment_name, "unset")
            for environment_name in THREAD_ENVIRONMENT_VARIABLES
        }
    )
    return metadata


def configure_cpu_runtime(number_of_cpus: int, log_metadata: bool = True) -> None:
    """Match reconfigurable thread pools to the current CPU assignment.

    Intra-op and BLAS thread counts may change between sequential branches.
    PyTorch inter-op threads are configured only on the first call in each
    process because ``set_num_interop_threads`` is a one-shot API.
    """

    global _INTEROP_THREADS_CONFIGURED
    if number_of_cpus < 1:
        raise ValueError("The CPU thread count must be positive.")

    thread_count = str(number_of_cpus)
    os.environ["OMP_NUM_THREADS"] = thread_count
    os.environ["MKL_NUM_THREADS"] = thread_count
    os.environ["OPENBLAS_NUM_THREADS"] = thread_count
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    torch.set_num_threads(number_of_cpus)
    if not _INTEROP_THREADS_CONFIGURED:
        # Mark before calling: if this PyTorch build reports that parallel work
        # already started, retrying later can never succeed and may abort.
        _INTEROP_THREADS_CONFIGURED = True
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError as error:
            warning("Could not set PyTorch inter-op threads: %s", error)

    if log_metadata:
        for key, value in cpu_runtime_metadata(number_of_cpus).items():
            info("CPU runtime %s: %s", key, value)
        info("PyTorch backend configuration:\n%s", torch.__config__.show())
