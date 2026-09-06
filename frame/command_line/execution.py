from pathlib import Path
from typing import Optional, Union

from frame.cluster.cluster_config import ClusterConfig
from frame.config_handle import UserConfig
from frame.context.execution_context import ExecutionContext
from frame.file_structure import (
    CONFIGS_DIR,
    CONTAINER_PROJECT_ROOT,
    LOCAL_PROJECT_ROOT,
    PROJECT_NAME,
    path_as_in_container,
)
from frame.git_tools import COMMIT_HASH_ENVIRONMENT_VARIABLE
from frame.python_environment import (
    cvmfs_python_activation_command,
    singularity_uv_cache_directory_export_command,
    uv_cache_directory_export_command,
)


CACHE_CONTENTION_EXIT_STATUS = 75
CACHE_LOCK_TIMEOUT_SEC = 300


QSUB_SCRIPT_HEADER = """#!/bin/bash
#PBS -m n
#PBS -S /bin/bash
#PBS -j oe
#PBS -N {job_name}
#PBS -q {queue}
#PBS -l walltime={walltime}
#PBS -l ncpus={ncpus}
#PBS -l mem={memory}g
{gpu_line}{array_job_line}
"""

QSUB_ENV_SETUP = """
# Environment setup
echo "Job started at: $(date)"
JOB_STARTED_AT_SECONDS=$(date +%s)
echo "Running on host: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Current directory: $(pwd)"
{task_id_line}{cvmfs_python_activation_command}
{uv_cache_directory_export_command}
{singularity_uv_cache_directory_export_command}
{environment_activation_command}

set -eo pipefail

log_job_completion() {{
    job_exit_status=$?
    trap - EXIT
    set +e

    if declare -F release_sandbox >/dev/null; then
        release_sandbox
        sandbox_cleanup_status=$?
        if [ "$sandbox_cleanup_status" -ne 0 ]; then
            echo "WARNING: Sandbox cleanup failed with exit status $sandbox_cleanup_status"
        fi
    fi

    finished_at=$(date +%s)
    echo "Job finished at: $(date)"
    echo "Job elapsed seconds: $((finished_at - JOB_STARTED_AT_SECONDS))"
    echo "Job exit status: $job_exit_status"
    exit "$job_exit_status"
}}
trap log_job_completion EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"""

QSUB_COMPLETION = """
# Job completion
exit $?
"""

# Singularity execution options:
# --no-mount tmp reduces file-descriptor pressure across large PBS arrays.
# --cleanenv prevents the host Python environment from leaking into the container.

SINGULARITY_EXECUTION_LINES = r"""
# -----------------------------------------------------------------------------
# Discover the resources that PBS actually exposed to this process.
# -----------------------------------------------------------------------------
REQUESTED_CPUS={ncpus}

detect_thread_count() {{
    local thread_count
    thread_count=$(nproc 2>/dev/null || true)
    if [ -n "$thread_count" ] && [ "$thread_count" -ge 1 ]; then
        echo "$thread_count"
        return 0
    fi

    # Affinity lists contain individual CPUs ("40") and ranges ("0-3,8-11").
    # Expand both forms when coreutils' nproc is unavailable.
    taskset -pc $$ 2>/dev/null | awk -F: '
        {{
            cpu_count = 0
            affinity_item_count = split($2, affinity_items, ",")
            for (item_index = 1; item_index <= affinity_item_count; item_index++) {{
                gsub(
                    /^[[:space:]]+|[[:space:]]+$/,
                    "",
                    affinity_items[item_index]
                )
                range_size = split(affinity_items[item_index], cpu_range, "-")
                cpu_count += range_size == 2 ? cpu_range[2] - cpu_range[1] + 1 : 1
            }}
            print cpu_count
        }}'
}}

detect_allocated_gpu_ids() {{
    if [ -n "${{CUDA_VISIBLE_DEVICES:-}}" ]; then
        echo "$CUDA_VISIBLE_DEVICES"
    elif [ -n "${{PBS_GPUFILE:-}}" ] && [ -r "$PBS_GPUFILE" ]; then
        awk 'NF {{print $NF}}' "$PBS_GPUFILE" | paste -sd, -
    fi
}}

THREADS_PER_PROCESS=$(detect_thread_count)
if [ -z "$THREADS_PER_PROCESS" ] || [ "$THREADS_PER_PROCESS" -lt 1 ]; then
    echo "ERROR: Could not determine the CPUs exposed to this job."
    exit 1
fi

HOST_CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-}}"
ALLOCATED_GPU_IDS=$(detect_allocated_gpu_ids)

# -----------------------------------------------------------------------------
# Pass scheduler state and threading limits into both container runtimes.
# -----------------------------------------------------------------------------
export_container_variable() {{
    local variable_name="$1"
    local variable_value="$2"
    export "SINGULARITYENV_${{variable_name}}=$variable_value"
    export "APPTAINERENV_${{variable_name}}=$variable_value"
}}

configure_container_environment() {{
    local passthrough_name
    local passthrough_value

    export_container_variable LFVDDP_ALLOCATED_CPUS "$THREADS_PER_PROCESS"
    export_container_variable LFVDDP_ALLOCATED_GPU_IDS "$ALLOCATED_GPU_IDS"
    if [ -n "$HOST_CUDA_VISIBLE_DEVICES" ]; then
        export_container_variable CUDA_VISIBLE_DEVICES "$HOST_CUDA_VISIBLE_DEVICES"
    fi

    for passthrough_name in PBS_JOBID PBS_ARRAY_INDEX PBS_NCPUS PBS_GPUFILE; do
        # Bash indirect expansion reads the variable named by passthrough_name.
        passthrough_value="${{!passthrough_name:-}}"
        export_container_variable "$passthrough_name" "$passthrough_value"
    done

    export_container_variable OMP_NUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable OMP_THREAD_LIMIT "$THREADS_PER_PROCESS"
    export_container_variable MKL_NUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable OPENBLAS_NUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable NUMEXPR_NUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable VECLIB_MAXIMUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable BLIS_NUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable TF_NUM_INTRAOP_THREADS "$THREADS_PER_PROCESS"
    export_container_variable TF_NUM_INTEROP_THREADS 1
    export_container_variable OMP_DYNAMIC FALSE
    export_container_variable MKL_DYNAMIC FALSE
    export_container_variable PYTHONUNBUFFERED 1
    export_container_variable PYTHONFAULTHANDLER 1
    export_container_variable {commit_hash_environment_variable} "{commit_hash}"
}}

log_runtime_diagnostics() {{
    local cgroup_file

    echo "Requested CPUs: $REQUESTED_CPUS"
    echo "Effective CPUs passed to training: $THREADS_PER_PROCESS"
    echo "Scheduler-assigned GPU IDs: ${{ALLOCATED_GPU_IDS:-none}}"
    echo "PBS job ID: ${{PBS_JOBID:-unavailable}}"
    echo "PBS array ID: ${{PBS_ARRAYID:-${{PBS_ARRAY_INDEX:-unavailable}}}}"
    echo "Host-visible CPUs: $(nproc 2>/dev/null || true)"
    echo "Process affinity: $(taskset -pc $$ 2>/dev/null || true)"

    if [ -n "${{PBS_NODEFILE:-}}" ] && [ -r "$PBS_NODEFILE" ]; then
        echo "PBS node file: $PBS_NODEFILE"
        echo "PBS allocated slots: $(wc -l < "$PBS_NODEFILE")"
        echo "PBS node file contents:"
        cat "$PBS_NODEFILE"
        echo "PBS slots grouped by host:"
        sort "$PBS_NODEFILE" | uniq -c || true
    fi

    echo "CPU topology:"
    lscpu 2>/dev/null || true
    echo "Host GPU diagnostics (not used for allocation):"
    nvidia-smi \
        --query-gpu=index,uuid,name,memory.total \
        --format=csv,noheader \
        2>/dev/null || echo "No host CUDA devices reported"

    for cgroup_file in \
        /sys/fs/cgroup/cpuset.cpus.effective \
        /sys/fs/cgroup/cpu.max \
        /sys/fs/cgroup/cpuset/cpuset.cpus \
        /sys/fs/cgroup/cpu/cpu.cfs_quota_us \
        /sys/fs/cgroup/cpu/cpu.cfs_period_us; do
        if [ -r "$cgroup_file" ]; then
            echo "$cgroup_file: $(tr '\n' ' ' < "$cgroup_file")"
        fi
    done
}}

configure_container_environment
log_runtime_diagnostics

# -----------------------------------------------------------------------------
# Prepare the immutable, node-local Singularity sandbox cache.
# -----------------------------------------------------------------------------
CONTAINER_SIF_PATH="{container_path}"
CACHE_ROOT="${{SINGULARITY_NODE_CACHE_DIR:-/tmp/$USER/singularity-node-cache}}"
mkdir -p "$CACHE_ROOT"

IMAGE_BASENAME=$(basename "$CONTAINER_SIF_PATH")
if ! IMAGE_ID=$(stat -Lc '%i.%s.%Y' "$CONTAINER_SIF_PATH"); then
    echo "ERROR: Container image is missing or inaccessible: $CONTAINER_SIF_PATH"
    exit 1
fi

# The inode, size, and modification time make each image revision a separate
# cache entry without hashing a potentially large SIF on every job start.
SANDBOX_KEY="${{IMAGE_BASENAME}}.${{IMAGE_ID}}"
SANDBOX_DIR="${{CACHE_ROOT}}/${{SANDBOX_KEY}}.sandbox"
READY_FILE="${{SANDBOX_DIR}}/.ready"

# Older scripts used a mkdir lock ending in .lock. A new .flock path ensures an
# abandoned legacy directory cannot block jobs after this upgrade.
LOCK_FILE="${{SANDBOX_DIR}}.flock"
LOCK_TIMEOUT_SEC="${{SINGULARITY_CACHE_LOCK_TIMEOUT_SEC:-{cache_lock_timeout_sec}}}"
CACHE_CONTENTION_EXIT_STATUS={cache_contention_exit_status}
CACHE_LOCK_HELD=0

LEASES_DIR="${{SANDBOX_DIR}}.leases"
LEASE_HOST=$(hostname)
LEASE_FILE="${{LEASES_DIR}}/${{LEASE_HOST}}.$$"

if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required for safe concurrent sandbox caching."
    exit 1
fi
# Keep one descriptor open for the job lifetime. The kernel releases its lock
# automatically when the owning process (or an in-progress build child) exits.
if ! exec {{CACHE_LOCK_FD}}>"$LOCK_FILE"; then
    echo "ERROR: Could not open sandbox cache lock: $LOCK_FILE"
    exit 1
fi

run_singularity() {{
    # The cluster injects a host-only I/O throttling library through LD_PRELOAD.
    # Singularity cannot mount that library while converting the SIF to a sandbox.
    (
        unset LD_PRELOAD
        {singularity_executable} "$@"
    )
}}

build_sandbox() {{
    local build_root
    local temporary_sandbox

    build_root=$(mktemp -d "${{CACHE_ROOT}}/.${{SANDBOX_KEY}}.build.XXXXXX")
    temporary_sandbox="${{build_root}}/sandbox"

    if ! run_singularity build --sandbox "$temporary_sandbox" "$CONTAINER_SIF_PATH"; then
        rm -rf "$build_root"
        return 1
    fi

    # Publish the ready marker and sandbox together via one atomic rename. A
    # published sandbox is immutable for the remainder of its lease lifetime.
    if ! touch "${{temporary_sandbox}}/.ready"; then
        rm -rf "$build_root"
        return 1
    fi
    if [ -e "$SANDBOX_DIR" ]; then
        if [ -f "$READY_FILE" ]; then
            rm -rf "$build_root"
            return 0
        fi
        echo "ERROR: Refusing to replace incomplete sandbox cache: $SANDBOX_DIR"
        rm -rf "$build_root"
        return 1
    fi
    if ! mv "$temporary_sandbox" "$SANDBOX_DIR"; then
        rm -rf "$build_root"
        return 1
    fi

    rmdir "$build_root" || true
}}

acquire_cache_lock() {{
    if [ "$CACHE_LOCK_HELD" -eq 1 ]; then
        return 0
    fi
    if ! flock -w "$LOCK_TIMEOUT_SEC" "$CACHE_LOCK_FD"; then
        return 1
    fi
    CACHE_LOCK_HELD=1
}}

release_cache_lock() {{
    if [ "$CACHE_LOCK_HELD" -eq 0 ]; then
        return 0
    fi
    if ! flock -u "$CACHE_LOCK_FD"; then
        return 1
    fi
    CACHE_LOCK_HELD=0
}}

acquire_sandbox_lease() {{
    local lease_status=1

    if ! acquire_cache_lock; then
        # Distinguish scheduler-level contention from a failed sandbox build.
        return "$CACHE_CONTENTION_EXIT_STATUS"
    fi

    if [ -f "$READY_FILE" ] || build_sandbox; then
        if mkdir -p "$LEASES_DIR" && touch "$LEASE_FILE"; then
            lease_status=0
        fi
    fi
    if ! release_cache_lock; then
        return 1
    fi
    return "$lease_status"
}}

yield_allocation_for_cache_contention() {{
    echo "Sandbox cache is busy after ${{LOCK_TIMEOUT_SEC}}s: $SANDBOX_DIR"
    # Exiting releases this job's allocation immediately. No automatic requeue
    # or resubmission is attempted because those require site-level PBS support.
    echo "Exiting with cache-contention status $CACHE_CONTENTION_EXIT_STATUS."
    exit "$CACHE_CONTENTION_EXIT_STATUS"
}}

release_sandbox() {{
    # A signal can arrive while the sandbox is still being built, before this
    # job owns a lease. Release any lock already held in that case.
    if [ ! -f "$LEASE_FILE" ]; then
        release_cache_lock
        return $?
    fi
    if [ "$CACHE_LOCK_HELD" -eq 0 ]; then
        if ! acquire_cache_lock; then
            echo "WARNING: Timed out waiting to clean sandbox: $SANDBOX_DIR"
            return 1
        fi
    fi

    rm -f "$LEASE_FILE"
    # Leases are node-local. Remove files whose process no longer exists on
    # this host, while preserving leases held by other live array jobs.
    for lease_candidate in "${{LEASES_DIR}}/${{LEASE_HOST}}."*; do
        [ -e "$lease_candidate" ] || continue
        lease_pid="${{lease_candidate##*.}}"
        if ! kill -0 "$lease_pid" 2>/dev/null; then
            rm -f "$lease_candidate"
        fi
    done

    # The immutable sandbox can be removed only after its final lease is gone.
    if [ -z "$(find "$LEASES_DIR" -type f -print -quit 2>/dev/null)" ]; then
        rm -rf "$SANDBOX_DIR" "$LEASES_DIR"
        echo "Removed unused sandbox cache: $SANDBOX_DIR"
    fi
    release_cache_lock
}}

# -----------------------------------------------------------------------------
# Acquire a sandbox lease, then run the requested command in that sandbox.
# -----------------------------------------------------------------------------
SANDBOX_ACQUIRE_STATUS=0
acquire_sandbox_lease || SANDBOX_ACQUIRE_STATUS=$?
case "$SANDBOX_ACQUIRE_STATUS" in
    0)
        ;;
    "$CACHE_CONTENTION_EXIT_STATUS")
        yield_allocation_for_cache_contention
        ;;
    *)
        echo "ERROR: Failed to prepare sandbox cache: $SANDBOX_DIR"
        exit 1
        ;;
esac

echo "Singularity executable: {singularity_executable}"
run_singularity --version || true
echo "Container SIF path: $CONTAINER_SIF_PATH"
echo "Sandbox cache root: $CACHE_ROOT"

if [ ! -f "$READY_FILE" ] || [ ! -f "$LEASE_FILE" ]; then
    echo "ERROR: Sandbox cache preparation did not produce a valid lease: $SANDBOX_DIR"
    exit 1
fi

run_singularity exec {gpu_passthrough_flag} \
    --no-mount tmp \
    --cleanenv \
    --pwd {container_project_root} \
    --bind {singularity_bindings} \
    "$SANDBOX_DIR" \
    {command}
"""


def _format_singularity_bindings(context: ExecutionContext) -> str:
    config = context.config
    bindings = [
        f"{Path(local_path).absolute()}:{container_path}"
        for local_path, container_path in config.config__bind_directories.items()
    ]
    bindings.append(
        f"{context.unique_out_dir.absolute()}:"
        f"{path_as_in_container(Path(config.config__out_dir).absolute())}"
    )
    return ",".join(bindings)


def format_qsub_execution_script(
    context: ExecutionContext,
    command: str,
    array_jobs: Optional[int] = None,
    use_gpu_if_needed: bool = True,
) -> str:
    config: ClusterConfig = context.config

    gpu_line = ""
    gpu_passthrough_flag = ""
    if use_gpu_if_needed and config.cluster__qsub_ngpus_for_train:
        gpu_line = f"#PBS -l ngpus={config.cluster__qsub_ngpus_for_train}\n"
        gpu_passthrough_flag = "--nv"

    return format_qsub_script(
        config=config,
        core_script_lines=SINGULARITY_EXECUTION_LINES,
        array_jobs=array_jobs,
        gpu_line=gpu_line,
        singularity_executable=config.cluster__singularity_executable,
        container_project_root=CONTAINER_PROJECT_ROOT,
        singularity_bindings=_format_singularity_bindings(context),
        container_path=LOCAL_PROJECT_ROOT / f"{PROJECT_NAME}.sif",
        command=command,
        gpu_passthrough_flag=gpu_passthrough_flag,
        commit_hash_environment_variable=COMMIT_HASH_ENVIRONMENT_VARIABLE,
        commit_hash=context.commit_hash,
        cache_contention_exit_status=CACHE_CONTENTION_EXIT_STATUS,
        cache_lock_timeout_sec=CACHE_LOCK_TIMEOUT_SEC,
    )


# Singularity build script. A few comments:
# Build command flags:
# --remote: the only option that works on the ATLAS cluster. Produces a SIF file.

SINGULARITY_BUILD_LINES = """
# Build Singularity container with custom repository and branch
echo "Building Singularity container..."

# Create working directory for build
BUILD_DIR=$(mktemp -d)
cd $BUILD_DIR

# The cluster injects a host-only I/O throttling library through LD_PRELOAD.
# Singularity cannot mount that library while converting the SIF to a sandbox.
run_singularity() (
    unset LD_PRELOAD
    exec "$@"
)

# Copy definition file from source path (passed as environment variable)
echo "Copying {project_name}.def file from $LFVDDP_DEF_PATH..."
cp $LFVDDP_DEF_PATH ./{project_name}.def

# Customize the definition file with repository URL, branch, and commit hash
# The commit hash is added as a comment to bust Singularity's layer cache
sed -e "s|REPO_URL=.*|REPO_URL=\"{repo_url}\"|" \
    -e "s|BRANCH=.*|BRANCH=\"{git_branch}\"|" \
    -e "s|COMMIT_HASH=.*|COMMIT_HASH=\"{git_commit_hash}\"|" \
    -e "s|CONTAINER_PROJECT_ROOT=.*|CONTAINER_PROJECT_ROOT=\"{container_project_root}\"|" \
    -e "s|# Cache-busting commit: PLACEHOLDER|# Cache-busting commit: {git_commit_hash}|" \
    {project_name}.def > {project_name}-edit.def

# Build from the customized definition file
echo "Building container..."
rm -f {project_name}.sif || true
run_singularity {singularity_executable} build --remote {project_name}.sif {project_name}-edit.def

# Exercise the image with the cluster's runtime mounts and environment before
# publishing it. The definition's %test imports the training entry point.
echo "Validating container..."
run_singularity {singularity_executable} test --cleanenv {project_name}.sif

# Publish only a validated, completely copied image. A failed build, test, or
# copy leaves the previous working SIF untouched.
PUBLISH_TMP="$PBS_O_WORKDIR/.{project_name}.sif.$$"
rm -f "$PUBLISH_TMP"
cp {project_name}.sif "$PUBLISH_TMP"
mv -f "$PUBLISH_TMP" "$PBS_O_WORKDIR/{project_name}.sif"

# Cleanup build directory
cd $PBS_O_WORKDIR
rm -rf $BUILD_DIR
"""


def format_qsub_build_script(
    config: ClusterConfig,
    git_branch: str,
    git_commit_hash: str,
) -> str:
    return format_qsub_script(
        config=config,
        core_script_lines=SINGULARITY_BUILD_LINES,
        array_jobs=0,
        gpu_line="",
        git_branch=git_branch,
        git_commit_hash=git_commit_hash,
        repo_url=config.cluster__repo_url,
        repo_name=config.repo_name,
        container_configs_dir=path_as_in_container(CONFIGS_DIR),
        container_project_root=CONTAINER_PROJECT_ROOT,
        singularity_executable=config.cluster__singularity_executable,
        project_name=PROJECT_NAME,
    )


def format_qsub_script(
    config: Union[ClusterConfig, UserConfig],
    core_script_lines: str,
    array_jobs: Optional[int] = None,
    **additional_template_kwargs,
) -> str:
    script = wrap_lines_with_qsub_script(core_script_lines)
    
    # Handle array jobs
    array_job_line = ""
    task_id_line = ""
    if array_jobs and array_jobs > 1:
        array_job_line = f"#PBS -J 1-{array_jobs}\n"
        task_id_line = 'echo "Task ID: $PBS_ARRAY_INDEX"\n'
    
    return script.format(
        job_name=config.config__dirsafe_runtag,
        queue=config.cluster__qsub_queue,
        walltime=config.cluster__qsub_walltime,
        ncpus=config.cluster__qsub_ncpus,
        memory=config.cluster__qsub_mem or 2,
        array_job_line=array_job_line,
        task_id_line=task_id_line,
        cvmfs_python_activation_command=cvmfs_python_activation_command(),
        uv_cache_directory_export_command=uv_cache_directory_export_command(
            config.cluster__uv_cache_dir
        ),
        singularity_uv_cache_directory_export_command=(
            singularity_uv_cache_directory_export_command(config.cluster__uv_cache_dir)
        ),
        environment_activation_command=config.cluster__environment_activation_command,
        **additional_template_kwargs,
    )


def wrap_lines_with_qsub_script(
    lines: str,
) -> str:
    return QSUB_SCRIPT_HEADER + QSUB_ENV_SETUP + lines + QSUB_COMPLETION
