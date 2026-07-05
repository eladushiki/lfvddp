import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

import pytest

from frame.command_line.handle_args import create_config_from_paths
from frame.context.execution_context import ExecutionContext
from frame.file_structure import LOCAL_PROJECT_ROOT, PROJECT_NAME
from test.environment import CONFIG_ARGUMENTS, ConfigType

JOB_WAIT_TIMEOUT_SECONDS = 20 * 60
POLL_SECONDS = 5


def create_submit_config(
    config_paths: Mapping[ConfigType, Path],
    out_dir: Path,
):
    return create_config_from_paths(
        config_paths=[config_paths[config_type] for config_type, _ in CONFIG_ARGUMENTS],
        is_plot=False,
        out_dir=str(out_dir),
    )


def submit_command(
    config_paths: Mapping[ConfigType, Path],
    out_dir: Path,
    continue_training: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "train/submit_train.py",
    ]
    for config_type, argument in CONFIG_ARGUMENTS:
        command.extend([argument, str(config_paths[config_type])])

    command.extend(
        [
            "--debug",
            "--no-build",
            "--only-train",
            "--out-dir",
            str(out_dir),
        ]
    )
    if continue_training:
        command.append("--continue")
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
            "submit_train.py failed\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def load_submit_context(out_dir: Path, dirsafe_runtag: str) -> ExecutionContext:
    context = ExecutionContext.find_stamped_submit_context(
        LOCAL_PROJECT_ROOT / out_dir,
        dirsafe_runtag,
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
