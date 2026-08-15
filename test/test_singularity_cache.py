import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from frame.command_line.execution import (
    CACHE_CONTENTION_EXIT_STATUS,
    CACHE_CONTENTION_EXIT_STATUS_ENV,
    format_qsub_execution_script,
)
from frame.submit import submit_command
from test.test_runtime_resources import RESOURCE_CLUSTER_CONFIG


PBS_HOOK_PATH = "frame/cluster/pbs_hooks/requeue_cache_contention.py"


@pytest.mark.parametrize(
    "function_execution_context",
    [RESOURCE_CLUSTER_CONFIG],
    indirect=True,
)
def test_qsub_cache_lock_recovers_after_owner_is_killed(
    function_execution_context,
    tmp_path,
):
    if shutil.which("flock") is None:
        pytest.skip("flock is only available on the Linux cluster runtime")

    script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
    )
    lock_setup = "LOCK_FILE=" + script.split("LOCK_FILE=", 1)[1].split(
        "run_singularity()", 1
    )[0]
    lock_functions = "acquire_cache_lock()" + script.split(
        "acquire_cache_lock()", 1
    )[1].split("acquire_sandbox_lease()", 1)[0]
    lock_script = f"""
set -eo pipefail
SANDBOX_DIR="$1/sandbox"
SINGULARITY_CACHE_LOCK_TIMEOUT_SEC=1
{lock_setup}
{lock_functions}
if ! acquire_cache_lock; then
    exit 42
fi
echo acquired
if [ "${{2:-}}" = hold ]; then
    while :; do
        read -r -t 1 || true
    done
fi
"""

    # Old versions used this directory as the lock. It must not interfere with
    # the new lock after an upgrade, even when an old job left it behind.
    (tmp_path / "sandbox.lock").mkdir()
    holder = subprocess.Popen(
        ["bash", "-c", lock_script, "bash", str(tmp_path), "hold"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "acquired"
        blocked = subprocess.run(
            ["bash", "-c", lock_script, "bash", str(tmp_path)],
            text=True,
            capture_output=True,
            timeout=5,
        )
        assert blocked.returncode == 42
    finally:
        holder.kill()
        holder.wait(timeout=5)

    recovered = subprocess.run(
        ["bash", "-c", lock_script, "bash", str(tmp_path)],
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert recovered.returncode == 0
    assert recovered.stdout.strip() == "acquired"


@pytest.mark.parametrize(
    "function_execution_context",
    [RESOURCE_CLUSTER_CONFIG],
    indirect=True,
)
def test_qsub_cache_contention_exits_for_pbs_hook(
    function_execution_context,
):
    script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
    )
    requeue_function = "requeue_for_cache_contention()" + script.split(
        "requeue_for_cache_contention()", 1
    )[1].split("release_sandbox()", 1)[0]
    result = subprocess.run(
        [
            "bash",
            "-c",
            f"""
set -eo pipefail
LOCK_TIMEOUT_SEC=5
CACHE_CONTENTION_EXIT_STATUS={CACHE_CONTENTION_EXIT_STATUS}
SANDBOX_DIR=/tmp/sandbox
{requeue_function}
requeue_for_cache_contention
""",
        ],
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == CACHE_CONTENTION_EXIT_STATUS
    assert "PBS cache-contention status" in result.stdout


@pytest.mark.parametrize(
    "function_execution_context",
    [RESOURCE_CLUSTER_CONFIG],
    indirect=True,
)
def test_submitted_qsub_job_is_marked_for_cache_contention_hook(
    function_execution_context,
    monkeypatch,
):
    submitted = {}

    def capture_submission(**kwargs):
        submitted.update(kwargs)
        return "4845839.pbs"

    monkeypatch.setattr("frame.submit.qsub_a_script", capture_submission)

    submit_command(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
    )

    assert submitted["env_vars"] == {
        CACHE_CONTENTION_EXIT_STATUS_ENV: str(CACHE_CONTENTION_EXIT_STATUS),
    }


@pytest.mark.parametrize(
    ("variables", "exit_status", "should_rerun"),
    [
        (
            {CACHE_CONTENTION_EXIT_STATUS_ENV: str(CACHE_CONTENTION_EXIT_STATUS)},
            CACHE_CONTENTION_EXIT_STATUS,
            True,
        ),
        ({}, CACHE_CONTENTION_EXIT_STATUS, False),
        (
            {CACHE_CONTENTION_EXIT_STATUS_ENV: str(CACHE_CONTENTION_EXIT_STATUS)},
            1,
            False,
        ),
    ],
)
def test_pbs_cache_hook_only_requeues_marked_contention(
    monkeypatch,
    variables,
    exit_status,
    should_rerun,
):
    calls = []
    job = SimpleNamespace(
        id="4845839[54].pbs",
        Variable_List=variables,
        Exit_status=exit_status,
        in_ms_mom=lambda: True,
        rerun=lambda: calls.append("rerun"),
    )

    def reject(message):
        calls.extend(("reject", message))
        raise SystemExit

    event = SimpleNamespace(
        job=job,
        accept=lambda: calls.append("accept"),
        reject=reject,
    )
    fake_pbs = SimpleNamespace(
        event=lambda: event,
        logjobmsg=lambda job_id, message: calls.extend((job_id, message)),
    )
    monkeypatch.setitem(sys.modules, "pbs", fake_pbs)

    if should_rerun:
        with pytest.raises(SystemExit):
            runpy.run_path(PBS_HOOK_PATH)
    else:
        runpy.run_path(PBS_HOOK_PATH)

    assert ("rerun" in calls) is should_rerun
    assert ("reject" in calls) is should_rerun
    assert ("accept" in calls) is (not should_rerun)
