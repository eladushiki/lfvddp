import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import pytest

from frame.context.execution_context import ExecutionContext
from frame.file_structure import (
    LOCAL_PROJECT_ROOT,
    PROJECT_NAME,
    SUBMIT_TRAIN_SCRIPT_NAME,
    SUBMIT_TRAIN_SCRIPT_RELATIVE,
)


JOB_WAIT_TIMEOUT_SECONDS = 20 * 60
POLL_SECONDS = 5


def build_submit_command(
    context: ExecutionContext,
    out_dir: Optional[Path] = None,
    continue_from: Optional[Path] = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SUBMIT_TRAIN_SCRIPT_RELATIVE),
    ]
    if continue_from is not None:
        return [*command, "--continue", str(continue_from)]
    if out_dir is None:
        raise ValueError("Fresh submission commands require an output directory.")

    command.append("--configs")
    command.extend(str(path) for path in context.typed_config_paths.values())

    command.extend(
        [
            "--debug",
            "--only-train",
            "--out-dir",
            str(out_dir),
        ]
    )
    return command


def run_submit(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=LOCAL_PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{SUBMIT_TRAIN_SCRIPT_NAME} failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def load_submit_context(out_dir: Path, dirsafe_runtag: str) -> ExecutionContext:
    context = ExecutionContext.find_stamped_run_context(
        LOCAL_PROJECT_ROOT / out_dir,
        dirsafe_runtag,
        entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
        require_continuation=True,
    )
    assert context is not None, (
        f"No submit context found below {LOCAL_PROJECT_ROOT / out_dir}"
    )
    return context


def _qstat_job_token(job_id: str) -> str:
    return re.sub(r"\[.*\]$", "", job_id.split(".", 1)[0])


def _job_is_active(job_id: str) -> bool:
    result = subprocess.run(
        ["qstat", "-f", _qstat_job_token(job_id)],
        cwd=LOCAL_PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False

    states = re.findall(r"job_state\s*=\s*(\w+)", result.stdout)
    if states and all(state in {"C", "E", "F"} for state in states):
        return False
    return True


def wait_for_job_to_finish(job_id: str) -> None:
    deadline = time.monotonic() + JOB_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if not _job_is_active(job_id):
            return
        time.sleep(POLL_SECONDS)
    pytest.fail(f"Timed out waiting for qsub job {job_id} to finish")


def require_server_prerequisites() -> None:
    if shutil.which("qsub") is None:
        pytest.skip("qsub is not available")
    if shutil.which("qstat") is None:
        pytest.skip("qstat is not available")
    if not (LOCAL_PROJECT_ROOT / f"{PROJECT_NAME}.sif").exists():
        pytest.skip(
            f"{PROJECT_NAME}.sif is not available; run or provide a container build first"
        )
