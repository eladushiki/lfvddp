import shutil
import subprocess

import pytest

from frame.command_line.execution import (
    CACHE_CONTENTION_EXIT_STATUS,
    format_qsub_build_script,
    format_qsub_execution_script,
)
from test.test_runtime_resources import RESOURCE_CLUSTER_CONFIG


@pytest.mark.parametrize(
    "function_execution_context",
    [RESOURCE_CLUSTER_CONFIG],
    indirect=True,
)
def test_container_build_is_pinned_validated_and_atomically_published(
    function_execution_context,
):
    script = format_qsub_build_script(
        config=function_execution_context.config,
        git_branch="main",
        git_commit_hash="0123456789abcdef",
    )

    pin = 'COMMIT_HASH="0123456789abcdef"'
    build = "singularity build --remote lfvddp.sif lfvddp-edit.def"
    validate = "singularity test lfvddp.sif"
    copy = 'cp lfvddp.sif "$PUBLISH_TMP"'
    publish = 'mv -f "$PUBLISH_TMP" "$PBS_O_WORKDIR/lfvddp.sif"'

    assert pin in script
    assert script.index(build) < script.index(validate)
    assert script.index(validate) < script.index(copy)
    assert script.index(copy) < script.index(publish)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)


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
def test_last_sandbox_lease_removes_extracted_cache(
    function_execution_context,
    tmp_path,
):
    if shutil.which("flock") is None:
        pytest.skip("flock is only available on the Linux cluster runtime")

    script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
    )
    cache_setup = "LOCK_FILE=" + script.split("LOCK_FILE=", 1)[1].split(
        "run_singularity()", 1
    )[0]
    cache_functions = "acquire_cache_lock()" + script.split(
        "acquire_cache_lock()", 1
    )[1].split("# Acquire a sandbox lease", 1)[0]
    cleanup_script = f"""
set -eo pipefail
SANDBOX_DIR="$1/sandbox"
SINGULARITY_CACHE_LOCK_TIMEOUT_SEC=1
{cache_setup}
{cache_functions}
mkdir -p "$SANDBOX_DIR" "$LEASES_DIR"
touch "$SANDBOX_DIR/extracted-file" "$LEASE_FILE"
release_sandbox
test ! -e "$SANDBOX_DIR"
test ! -e "$LEASES_DIR"
test -f "$LOCK_FILE"
"""

    subprocess.run(
        ["bash", "-c", cleanup_script, "bash", str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [RESOURCE_CLUSTER_CONFIG],
    indirect=True,
)
def test_qsub_cache_contention_exits_without_resubmitting(
    function_execution_context,
):
    script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py --continue run",
    )
    yield_function = "yield_allocation_for_cache_contention()" + script.split(
        "yield_allocation_for_cache_contention()", 1
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
{yield_function}
yield_allocation_for_cache_contention
""",
        ],
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == CACHE_CONTENTION_EXIT_STATUS
    assert "cache-contention status" in result.stdout
    assert "qrerun" not in yield_function
    assert "qsub" not in yield_function
