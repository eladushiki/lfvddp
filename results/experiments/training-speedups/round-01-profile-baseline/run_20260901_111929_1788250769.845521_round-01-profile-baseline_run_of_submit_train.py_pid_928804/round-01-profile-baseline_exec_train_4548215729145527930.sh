#!/bin/bash
#PBS -m n
#PBS -S /bin/bash
#PBS -j oe
#PBS -N round-01-profile-baseline
#PBS -q N
#PBS -l walltime=12:00:00
#PBS -l ncpus=8
#PBS -l mem=2g


# Environment setup
echo "Job started at: $(date)"
JOB_STARTED_AT_SECONDS=$(date +%s)
echo "Running on host: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Current directory: $(pwd)"
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh
export UV_CACHE_DIR=/tmp/lfvddp-uv-cache
export SINGULARITYENV_UV_CACHE_DIR=/tmp/lfvddp-uv-cache
export PATH="/cvmfs/atlas.cern.ch/repo/containers/sw/singularity/x86_64-el9/current/bin:$PATH"; export SINGULARITY_BINDPATH="/storage/agrp/eladklig/SymmetrizedDDP/.venv:/app/.venv${SINGULARITY_BINDPATH:+:$SINGULARITY_BINDPATH}"

set -eo pipefail

log_job_completion() {
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
}
trap log_job_completion EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# -----------------------------------------------------------------------------
# Discover the resources that PBS actually exposed to this process.
# -----------------------------------------------------------------------------
REQUESTED_CPUS=8

detect_thread_count() {
    local thread_count
    thread_count=$(nproc 2>/dev/null || true)
    if [ -n "$thread_count" ] && [ "$thread_count" -ge 1 ]; then
        echo "$thread_count"
        return 0
    fi

    # Affinity lists contain individual CPUs ("40") and ranges ("0-3,8-11").
    # Expand both forms when coreutils' nproc is unavailable.
    taskset -pc $$ 2>/dev/null | awk -F: '
        {
            cpu_count = 0
            affinity_item_count = split($2, affinity_items, ",")
            for (item_index = 1; item_index <= affinity_item_count; item_index++) {
                gsub(
                    /^[[:space:]]+|[[:space:]]+$/,
                    "",
                    affinity_items[item_index]
                )
                range_size = split(affinity_items[item_index], cpu_range, "-")
                cpu_count += range_size == 2 ? cpu_range[2] - cpu_range[1] + 1 : 1
            }
            print cpu_count
        }'
}

detect_allocated_gpu_ids() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "$CUDA_VISIBLE_DEVICES"
    elif [ -n "${PBS_GPUFILE:-}" ] && [ -r "$PBS_GPUFILE" ]; then
        awk 'NF {print $NF}' "$PBS_GPUFILE" | paste -sd, -
    fi
}

THREADS_PER_PROCESS=$(detect_thread_count)
if [ -z "$THREADS_PER_PROCESS" ] || [ "$THREADS_PER_PROCESS" -lt 1 ]; then
    echo "ERROR: Could not determine the CPUs exposed to this job."
    exit 1
fi

HOST_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
ALLOCATED_GPU_IDS=$(detect_allocated_gpu_ids)

# -----------------------------------------------------------------------------
# Pass scheduler state and threading limits into both container runtimes.
# -----------------------------------------------------------------------------
export_container_variable() {
    local variable_name="$1"
    local variable_value="$2"
    export "SINGULARITYENV_${variable_name}=$variable_value"
    export "APPTAINERENV_${variable_name}=$variable_value"
}

configure_container_environment() {
    local passthrough_name
    local passthrough_value

    export_container_variable LFVDDP_ALLOCATED_CPUS "$THREADS_PER_PROCESS"
    export_container_variable LFVDDP_ALLOCATED_GPU_IDS "$ALLOCATED_GPU_IDS"
    if [ -n "$HOST_CUDA_VISIBLE_DEVICES" ]; then
        export_container_variable CUDA_VISIBLE_DEVICES "$HOST_CUDA_VISIBLE_DEVICES"
    fi

    for passthrough_name in PBS_JOBID PBS_ARRAY_INDEX PBS_NCPUS PBS_GPUFILE; do
        # Bash indirect expansion reads the variable named by passthrough_name.
        passthrough_value="${!passthrough_name:-}"
        export_container_variable "$passthrough_name" "$passthrough_value"
    done

    export_container_variable OMP_NUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable MKL_NUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable OPENBLAS_NUM_THREADS "$THREADS_PER_PROCESS"
    export_container_variable OMP_DYNAMIC FALSE
    export_container_variable MKL_DYNAMIC FALSE
    export_container_variable PYTHONUNBUFFERED 1
    export_container_variable PYTHONFAULTHANDLER 1
    export_container_variable LFVDDP_COMMIT_HASH "03bfa6a6944c1da6099ecaef6d4bb602419ec9f7"
}

log_runtime_diagnostics() {
    local cgroup_file

    echo "Requested CPUs: $REQUESTED_CPUS"
    echo "Effective CPUs passed to training: $THREADS_PER_PROCESS"
    echo "Scheduler-assigned GPU IDs: ${ALLOCATED_GPU_IDS:-none}"
    echo "PBS job ID: ${PBS_JOBID:-unavailable}"
    echo "PBS array ID: ${PBS_ARRAYID:-${PBS_ARRAY_INDEX:-unavailable}}"
    echo "Host-visible CPUs: $(nproc 2>/dev/null || true)"
    echo "Process affinity: $(taskset -pc $$ 2>/dev/null || true)"

    if [ -n "${PBS_NODEFILE:-}" ] && [ -r "$PBS_NODEFILE" ]; then
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
}

configure_container_environment
log_runtime_diagnostics

# -----------------------------------------------------------------------------
# Prepare the immutable, node-local Singularity sandbox cache.
# -----------------------------------------------------------------------------
CONTAINER_SIF_PATH="/storage/agrp/eladklig/SymmetrizedDDP/lfvddp.sif"
CACHE_ROOT="${SINGULARITY_NODE_CACHE_DIR:-/tmp/$USER/singularity-node-cache}"
mkdir -p "$CACHE_ROOT"

IMAGE_BASENAME=$(basename "$CONTAINER_SIF_PATH")
if ! IMAGE_ID=$(stat -Lc '%i.%s.%Y' "$CONTAINER_SIF_PATH"); then
    echo "ERROR: Container image is missing or inaccessible: $CONTAINER_SIF_PATH"
    exit 1
fi

# The inode, size, and modification time make each image revision a separate
# cache entry without hashing a potentially large SIF on every job start.
SANDBOX_KEY="${IMAGE_BASENAME}.${IMAGE_ID}"
SANDBOX_DIR="${CACHE_ROOT}/${SANDBOX_KEY}.sandbox"
READY_FILE="${SANDBOX_DIR}/.ready"

# Older scripts used a mkdir lock ending in .lock. A new .flock path ensures an
# abandoned legacy directory cannot block jobs after this upgrade.
LOCK_FILE="${SANDBOX_DIR}.flock"
LOCK_TIMEOUT_SEC="${SINGULARITY_CACHE_LOCK_TIMEOUT_SEC:-300}"
CACHE_CONTENTION_EXIT_STATUS=75
CACHE_LOCK_HELD=0

LEASES_DIR="${SANDBOX_DIR}.leases"
LEASE_HOST=$(hostname)
LEASE_FILE="${LEASES_DIR}/${LEASE_HOST}.$$"

if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required for safe concurrent sandbox caching."
    exit 1
fi
# Keep one descriptor open for the job lifetime. The kernel releases its lock
# automatically when the owning process (or an in-progress build child) exits.
if ! exec {CACHE_LOCK_FD}>"$LOCK_FILE"; then
    echo "ERROR: Could not open sandbox cache lock: $LOCK_FILE"
    exit 1
fi

run_singularity() {
    # The cluster injects a host-only I/O throttling library through LD_PRELOAD.
    # Singularity cannot mount that library while converting the SIF to a sandbox.
    (
        unset LD_PRELOAD
        singularity "$@"
    )
}

build_sandbox() {
    local build_root
    local temporary_sandbox

    build_root=$(mktemp -d "${CACHE_ROOT}/.${SANDBOX_KEY}.build.XXXXXX")
    temporary_sandbox="${build_root}/sandbox"

    if ! run_singularity build --sandbox "$temporary_sandbox" "$CONTAINER_SIF_PATH"; then
        rm -rf "$build_root"
        return 1
    fi

    # Publish the ready marker and sandbox together via one atomic rename. A
    # published sandbox is immutable for the remainder of its lease lifetime.
    if ! touch "${temporary_sandbox}/.ready"; then
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
}

acquire_cache_lock() {
    if [ "$CACHE_LOCK_HELD" -eq 1 ]; then
        return 0
    fi
    if ! flock -w "$LOCK_TIMEOUT_SEC" "$CACHE_LOCK_FD"; then
        return 1
    fi
    CACHE_LOCK_HELD=1
}

release_cache_lock() {
    if [ "$CACHE_LOCK_HELD" -eq 0 ]; then
        return 0
    fi
    if ! flock -u "$CACHE_LOCK_FD"; then
        return 1
    fi
    CACHE_LOCK_HELD=0
}

acquire_sandbox_lease() {
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
}

yield_allocation_for_cache_contention() {
    echo "Sandbox cache is busy after ${LOCK_TIMEOUT_SEC}s: $SANDBOX_DIR"
    # Exiting releases this job's allocation immediately. No automatic requeue
    # or resubmission is attempted because those require site-level PBS support.
    echo "Exiting with cache-contention status $CACHE_CONTENTION_EXIT_STATUS."
    exit "$CACHE_CONTENTION_EXIT_STATUS"
}

release_sandbox() {
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
    for lease_candidate in "${LEASES_DIR}/${LEASE_HOST}."*; do
        [ -e "$lease_candidate" ] || continue
        lease_pid="${lease_candidate##*.}"
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
}

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

echo "Singularity executable: singularity"
run_singularity --version || true
echo "Container SIF path: $CONTAINER_SIF_PATH"
echo "Sandbox cache root: $CACHE_ROOT"

if [ ! -f "$READY_FILE" ] || [ ! -f "$LEASE_FILE" ]; then
    echo "ERROR: Sandbox cache preparation did not produce a valid lease: $SANDBOX_DIR"
    exit 1
fi

run_singularity exec  \
    --no-mount tmp \
    --cleanenv \
    --pwd /app \
    --bind /storage/agrp/eladklig/SymmetrizedDDP/.worktrees/training-speedups:/app,/cvmfs:/cvmfs/,/storage/agrp/eladklig/SymmetrizedDDP/.worktrees/training-speedups/results/experiments/training-speedups/round-01-profile-baseline/run_20260901_111929_1788250769.845521_round-01-profile-baseline_run_of_submit_train.py_pid_928804:/app/.worktrees/training-speedups/results/experiments/training-speedups/round-01-profile-baseline \
    "$SANDBOX_DIR" \
    python train/single_train.py --configs /app/.worktrees/training-speedups/results/experiments/training-speedups/round-01-profile-baseline/configs --only-train

# Job completion
exit $?
