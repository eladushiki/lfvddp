from pathlib import Path
import subprocess

import pytest

from frame.command_line.execution import format_qsub_execution_script
from train.thread_probe import (
    THREAD_PROBE_CASES,
    THREAD_PROBE_ENABLED_ENV,
    format_thread_probe_case_setup,
    format_thread_probe_monitor_functions,
)
from train.thread_probe_monitor import sample_session


def test_thread_probe_requires_its_six_element_array(monkeypatch):
    monkeypatch.setenv(THREAD_PROBE_ENABLED_ENV, "1")

    with pytest.raises(ValueError, match="exactly 6"):
        format_thread_probe_case_setup(5)


@pytest.mark.parametrize("function_execution_context", [{}], indirect=True)
def test_thread_probe_is_embedded_in_six_element_qsub_script(
    function_execution_context,
    monkeypatch,
):
    monkeypatch.setenv(THREAD_PROBE_ENABLED_ENV, "1")

    script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
        array_jobs=len(THREAD_PROBE_CASES),
        use_gpu_if_needed=False,
    )

    assert "#PBS -J 1-6" in script
    assert 'THREAD_PROBE_CASE_NAME="baseline"' in script
    assert (
        'export_container_variable LFVDDP_THREAD_PROBE_CASE '
        '"$THREAD_PROBE_CASE_NAME"' in script
    )
    assert 'export_container_variable LFVDDP_PROBE_OPENBLAS_THREADS "1"' in script
    assert 'export_container_variable LFVDDP_PROBE_OMP_THREADS "1"' in script
    assert 'export_container_variable LFVDDP_PROBE_TORCH_CAPACITY "2"' in script
    assert 'export_container_variable LFVDDP_PROBE_FORCE_SEQUENTIAL "1"' in script
    assert "start_thread_probe_monitor" in script
    assert '"$PBS_O_WORKDIR/train/thread_probe_monitor.py"' in script
    subprocess.run(
        ["bash", "-n"],
        input=(
            "export_container_variable() { :; }\n"
            + format_thread_probe_case_setup(len(THREAD_PROBE_CASES))
            + format_thread_probe_monitor_functions()
        ),
        text=True,
        check=True,
    )


@pytest.mark.parametrize("function_execution_context", [{}], indirect=True)
def test_thread_probe_is_absent_by_default(function_execution_context, monkeypatch):
    monkeypatch.delenv(THREAD_PROBE_ENABLED_ENV, raising=False)

    script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
        array_jobs=6,
        use_gpu_if_needed=False,
    )

    assert "THREAD_PROBE_ENABLED=1" not in script
    assert "start_thread_probe_monitor" not in script
    assert "train/thread_probe_monitor.py" not in script


def _write_stat(path: Path, pid: int, comm: str, state: str, session: int) -> None:
    path.write_text(f"{pid} ({comm}) {state} 1 {pid} {session} 0 0 0\n")


def _write_process(
    proc_root: Path,
    pid: int,
    session: int,
    states: tuple[str, ...],
) -> None:
    process_dir = proc_root / str(pid)
    task_dir = process_dir / "task"
    task_dir.mkdir(parents=True)
    _write_stat(process_dir / "stat", pid, "python worker", states[0], session)
    for offset, state in enumerate(states):
        tid = pid + offset
        thread_dir = task_dir / str(tid)
        thread_dir.mkdir()
        _write_stat(thread_dir / "stat", tid, "python worker", state, session)


def test_session_sampler_counts_tasks_and_excludes_monitor(tmp_path):
    _write_process(tmp_path, 100, 42, ("S", "R", "S"))
    _write_process(tmp_path, 200, 42, ("R",))
    _write_process(tmp_path, 300, 99, ("R", "R"))

    sample = sample_session(42, excluded_pid=200, proc_root=tmp_path)

    assert sample["threads"] == 3
    assert sample["runnable"] == 1
    assert sample["sleeping"] == 2
    assert sample["other"] == 0
    assert sample["states"] == {"R": 1, "S": 2}
    assert sample["processes"] == [
        {
            "pid": 100,
            "comm": "python worker",
            "threads": 3,
            "runnable": 1,
            "states": {"R": 1, "S": 2},
            "thread_names": {"python worker": 3},
        }
    ]
