"""Temporary six-case thread-pool experiment configuration.

The probe is opt-in at submission time through ``LFVDDP_THREAD_PROBE=1``.
Normal training never reads the array index and retains its existing resource
policy.
"""

from dataclasses import dataclass
import os
from typing import Mapping, Optional


THREAD_PROBE_ENABLED_ENV = "LFVDDP_THREAD_PROBE"
THREAD_PROBE_CASE_ENV = "LFVDDP_THREAD_PROBE_CASE"
PROBE_OMP_THREADS_ENV = "LFVDDP_PROBE_OMP_THREADS"
PROBE_OPENBLAS_THREADS_ENV = "LFVDDP_PROBE_OPENBLAS_THREADS"
PROBE_TORCH_CAPACITY_ENV = "LFVDDP_PROBE_TORCH_CAPACITY"
PROBE_FORCE_SEQUENTIAL_ENV = "LFVDDP_PROBE_FORCE_SEQUENTIAL"


@dataclass(frozen=True)
class ThreadProbeCase:
    """One controlled combination in the temporary PBS array."""

    index: int
    name: str
    omp_threads: Optional[int] = None
    openblas_threads: Optional[int] = None
    torch_capacity: Optional[int] = None
    force_sequential: bool = False


THREAD_PROBE_CASES = (
    ThreadProbeCase(1, "baseline"),
    ThreadProbeCase(2, "openblas-1", openblas_threads=1),
    ThreadProbeCase(3, "openmp-1", omp_threads=1),
    ThreadProbeCase(4, "openmp-1-openblas-1", omp_threads=1, openblas_threads=1),
    ThreadProbeCase(
        5,
        "parallel-torch-1-plus-1",
        omp_threads=1,
        openblas_threads=1,
        torch_capacity=2,
    ),
    ThreadProbeCase(
        6,
        "sequential-torch-1",
        omp_threads=1,
        openblas_threads=1,
        torch_capacity=1,
        force_sequential=True,
    ),
)


def _positive_integer(environment: Mapping[str, str], name: str) -> Optional[int]:
    value = environment.get(name)
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return parsed


def probe_thread_limit(
    environment_name: str,
    default: int,
    environment: Mapping[str, str] = os.environ,
) -> int:
    """Return an opt-in startup-pool override or the normal runtime value."""

    probe_name_by_runtime_name = {
        "OMP_NUM_THREADS": PROBE_OMP_THREADS_ENV,
        "OPENBLAS_NUM_THREADS": PROBE_OPENBLAS_THREADS_ENV,
    }
    probe_name = probe_name_by_runtime_name.get(environment_name)
    if probe_name is None:
        return default
    return _positive_integer(environment, probe_name) or default


def probe_torch_capacity(
    default: int,
    environment: Mapping[str, str] = os.environ,
) -> int:
    """Return the temporary total Torch capacity when the probe requests one."""

    return _positive_integer(environment, PROBE_TORCH_CAPACITY_ENV) or default


def probe_forces_sequential(
    environment: Mapping[str, str] = os.environ,
) -> bool:
    """Return whether the final probe case requests direct parent execution."""

    return environment.get(PROBE_FORCE_SEQUENTIAL_ENV) == "1"


def _shell_assignment(name: str, value: object) -> str:
    return f'export_container_variable {name} "{value}"'


def format_thread_probe_case_setup(array_jobs: Optional[int]) -> str:
    """Render the one source-of-truth case table into the generated PBS script."""

    if os.environ.get(THREAD_PROBE_ENABLED_ENV) != "1":
        return ""
    if array_jobs != len(THREAD_PROBE_CASES):
        raise ValueError(
            "The thread probe requires exactly "
            f"{len(THREAD_PROBE_CASES)} PBS array elements."
        )

    case_blocks = []
    for case in THREAD_PROBE_CASES:
        assignments = [
            f'THREAD_PROBE_CASE_NAME="{case.name}"',
            _shell_assignment(THREAD_PROBE_CASE_ENV, "$THREAD_PROBE_CASE_NAME"),
        ]
        if case.omp_threads is not None:
            assignments.extend(
                (
                    f'PROBE_OMP_NUM_THREADS="{case.omp_threads}"',
                    _shell_assignment(PROBE_OMP_THREADS_ENV, case.omp_threads),
                )
            )
        if case.openblas_threads is not None:
            assignments.extend(
                (
                    f'PROBE_OPENBLAS_NUM_THREADS="{case.openblas_threads}"',
                    _shell_assignment(
                        PROBE_OPENBLAS_THREADS_ENV,
                        case.openblas_threads,
                    ),
                )
            )
        if case.torch_capacity is not None:
            assignments.append(
                _shell_assignment(PROBE_TORCH_CAPACITY_ENV, case.torch_capacity)
            )
        if case.force_sequential:
            assignments.append(_shell_assignment(PROBE_FORCE_SEQUENTIAL_ENV, 1))
        body = "\n            ".join(assignments)
        case_blocks.append(f"        {case.index})\n            {body}\n            ;;")

    cases = "\n".join(case_blocks)
    return f'''\nTHREAD_PROBE_ENABLED=1
PROBE_OMP_NUM_THREADS="$THREADS_PER_PROCESS"
PROBE_OPENBLAS_NUM_THREADS="$THREADS_PER_PROCESS"
case "${{PBS_ARRAY_INDEX:-}}" in
{cases}
        *)
            echo "ERROR: Thread probe requires PBS_ARRAY_INDEX=1-{len(THREAD_PROBE_CASES)}."
            exit 1
            ;;
esac
echo "Thread probe case: ${{PBS_ARRAY_INDEX}}"
'''


def format_thread_probe_monitor_functions() -> str:
    """Return lifecycle helpers only for probe-generated PBS scripts."""

    if os.environ.get(THREAD_PROBE_ENABLED_ENV) != "1":
        return ""
    return r'''
start_thread_probe_monitor() {
    python "$PBS_O_WORKDIR/train/thread_probe_monitor.py" \
        --case "$PBS_ARRAY_INDEX:$THREAD_PROBE_CASE_NAME" \
        --interval 0.1 &
    THREAD_PROBE_MONITOR_PID=$!
}

stop_thread_probe_monitor() {
    if [ -n "${THREAD_PROBE_MONITOR_PID:-}" ]; then
        kill -TERM "$THREAD_PROBE_MONITOR_PID" 2>/dev/null || true
        wait "$THREAD_PROBE_MONITOR_PID" || true
        THREAD_PROBE_MONITOR_PID=""
    fi
}
'''


def format_thread_probe_monitor_start() -> str:
    if os.environ.get(THREAD_PROBE_ENABLED_ENV) != "1":
        return ""
    return "start_thread_probe_monitor\n"


def format_thread_probe_cleanup() -> str:
    if os.environ.get(THREAD_PROBE_ENABLED_ENV) != "1":
        return ""
    return '''\n    if declare -F stop_thread_probe_monitor >/dev/null; then
        stop_thread_probe_monitor
    fi
'''
