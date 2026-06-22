from pathlib import Path
from typing import Optional, Union

from frame.cluster.cluster_config import ClusterConfig
from frame.config_handle import UserConfig
from frame.context.execution_context import ExecutionContext
from frame.file_structure import CONFIGS_DIR, CONTAINER_PROJECT_ROOT, LOCAL_PROJECT_ROOT, PROJECT_NAME, path_as_in_container


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
echo "Running on host: $(hostname)"
echo "Job ID: $PBS_JOBID"
echo "Current directory: $(pwd)"
{task_id_line}{environment_activation_command}

set -eo pipefail
"""

QSUB_COMPLETION = """
# Job completion
exit $?
"""

# Singularity exec command. A few comments:
# --no-mount tmp: relieves pressure on file descriptors when running many jobs in parallel
# --cleanenv: avoids messing with host environment, i.e. python stuff

SINGULARITY_EXECUTION_LINES = r"""
CONTAINER_SIF_PATH="{container_path}"
CACHE_ROOT="${{SINGULARITY_NODE_CACHE_DIR:-/tmp/$USER/singularity-node-cache}}"
mkdir -p "$CACHE_ROOT"

IMG_BASENAME=$(basename "$CONTAINER_SIF_PATH")
IMG_MTIME=$(stat -c %Y "$CONTAINER_SIF_PATH" 2>/dev/null || echo 0)
SANDBOX_KEY="${{IMG_BASENAME}}.${{IMG_MTIME}}"
SANDBOX_DIR="${{CACHE_ROOT}}/${{SANDBOX_KEY}}.sandbox"
READY_FILE="${{SANDBOX_DIR}}/.ready"
LOCK_DIR="${{SANDBOX_DIR}}.lock"
LOCK_TIMEOUT_SEC="${{SINGULARITY_CACHE_LOCK_TIMEOUT_SEC:-1800}}"

run_singularity() {{
    {singularity_executable} "$@"
}}

build_sandbox() {{
    TMP_SANDBOX="${{SANDBOX_DIR}}.tmp.$$"
    rm -rf "$TMP_SANDBOX"
    if run_singularity build --sandbox "$TMP_SANDBOX" "$CONTAINER_SIF_PATH"; then
        rm -rf "$SANDBOX_DIR"
        mv "$TMP_SANDBOX" "$SANDBOX_DIR"
        touch "$READY_FILE"
        return 0
    fi
    rm -rf "$TMP_SANDBOX"
    return 1
}}

if [ ! -f "$READY_FILE" ]; then
    SANDBOX_RETRY_MAX=${{SINGULARITY_SANDBOX_RETRY_MAX:-6}}
    SANDBOX_RETRY_BASE_SLEEP_SEC=${{SINGULARITY_SANDBOX_RETRY_BASE_SLEEP_SEC:-20}}
    SANDBOX_RETRY_JITTER_SEC=${{SINGULARITY_SANDBOX_RETRY_JITTER_SEC:-40}}
    SANDBOX_ATTEMPT=1

    while [ ! -f "$READY_FILE" ] && [ "$SANDBOX_ATTEMPT" -le "$SANDBOX_RETRY_MAX" ]; do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            build_sandbox || true
            rmdir "$LOCK_DIR" 2>/dev/null || true
        else
            START_WAIT=$(date +%s)
            while [ -d "$LOCK_DIR" ] && [ ! -f "$READY_FILE" ]; do
                NOW=$(date +%s)
                if [ $((NOW - START_WAIT)) -ge "$LOCK_TIMEOUT_SEC" ]; then
                    break
                fi
                sleep 2
            done
        fi

        if [ -f "$READY_FILE" ]; then
            break
        fi

        SLEEP_SEC=$((SANDBOX_RETRY_BASE_SLEEP_SEC + RANDOM % (SANDBOX_RETRY_JITTER_SEC + 1)))
        echo "Sandbox not ready (attempt $SANDBOX_ATTEMPT/$SANDBOX_RETRY_MAX). Retrying in ${{SLEEP_SEC}}s."
        sleep "$SLEEP_SEC"
        SANDBOX_ATTEMPT=$((SANDBOX_ATTEMPT + 1))
    done
fi

echo "Singularity executable: {singularity_executable}"
run_singularity --version || true
echo "Container SIF path: $CONTAINER_SIF_PATH"
echo "Sandbox cache root: $CACHE_ROOT"

if [ -f "$READY_FILE" ]; then
    CONTAINER_RUNTIME_PATH="$SANDBOX_DIR"
else
    echo "ERROR: sandbox not available at $SANDBOX_DIR after retries."
    exit 1
fi

TORCHINDUCTOR_HOST_CACHE_ROOT="${{TORCHINDUCTOR_NODE_CACHE_DIR:-/tmp/$USER/torchinductor-cache}}"
TORCHINDUCTOR_CONTAINER_CACHE_ROOT="/tmp/torchinductor-cache"
TORCHINDUCTOR_JOB_CACHE_KEY="${{PBS_JOBID:-manual.$$}}"
TORCHINDUCTOR_TASK_CACHE_KEY="${{PBS_ARRAY_INDEX:-0}}"
TORCHINDUCTOR_HOST_CACHE_DIR="${{TORCHINDUCTOR_HOST_CACHE_ROOT}}/${{TORCHINDUCTOR_JOB_CACHE_KEY}}/${{TORCHINDUCTOR_TASK_CACHE_KEY}}"
TORCHINDUCTOR_CONTAINER_CACHE_DIR="${{TORCHINDUCTOR_CONTAINER_CACHE_ROOT}}/${{TORCHINDUCTOR_JOB_CACHE_KEY}}/${{TORCHINDUCTOR_TASK_CACHE_KEY}}"
mkdir -p "$TORCHINDUCTOR_HOST_CACHE_DIR"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CONTAINER_CACHE_DIR"
echo "TorchInductor cache: $TORCHINDUCTOR_HOST_CACHE_DIR -> $TORCHINDUCTOR_CONTAINER_CACHE_DIR"

run_singularity exec --no-mount tmp --cleanenv --pwd {container_project_root} --bind {singularity_bindings},"${{TORCHINDUCTOR_HOST_CACHE_ROOT}}:${{TORCHINDUCTOR_CONTAINER_CACHE_ROOT}}" "$CONTAINER_RUNTIME_PATH" {command}
"""


def format_qsub_execution_script(
        context: ExecutionContext,
        command: str,
        array_jobs: Optional[int] = None,
        use_gpu_if_needed: bool = True,
    ) -> str:
    config: ClusterConfig = context.config

    # Handle GPU line
    gpu_line = ""
    if use_gpu_if_needed and config.cluster__qsub_ngpus_for_train:
        gpu_line = f"#PBS -l ngpus={config.cluster__qsub_ngpus_for_train}\n"

    singularity_bindings = ",".join([
        f"{Path(local_path).absolute()}:{container_path}"
        for local_path, container_path in context.config.config__bind_directories.items()
    ] + [f"{context.unique_out_dir.absolute()}:{path_as_in_container(Path(config.config__out_dir).absolute())}"])

    return format_qsub_script(
        config=config,
        core_script_lines=SINGULARITY_EXECUTION_LINES,
        array_jobs=array_jobs,
        gpu_line=gpu_line,
        singularity_executable=config.cluster__singularity_executable,
        container_project_root=CONTAINER_PROJECT_ROOT,
        singularity_bindings=singularity_bindings,
        container_path=LOCAL_PROJECT_ROOT / f"{PROJECT_NAME}.sif",
        command=command,
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

# Copy definition file from source path (passed as environment variable)
echo "Copying {project_name}.def file from $LFVDDP_DEF_PATH..."
cp $LFVDDP_DEF_PATH ./{project_name}.def

# Customize the definition file with repository URL, branch, and commit hash
# The commit hash is added as a comment to bust Singularity's layer cache
sed -e "s|REPO_URL=.*|REPO_URL=\"{repo_url}\"|" \
    -e "s|BRANCH=.*|BRANCH=\"{git_branch}\"|" \
    -e "s|CONTAINER_PROJECT_ROOT=.*|CONTAINER_PROJECT_ROOT=\"{container_project_root}\"|" \
    -e "s|# Cache-busting commit: PLACEHOLDER|# Cache-busting commit: {git_commit_hash}|" \
    {project_name}.def > {project_name}-edit.def

# Build from the customized definition file
echo "Building container..."
rm -f {project_name}.sif || true
{singularity_executable} build --remote {project_name}.sif {project_name}-edit.def

# Copy the built SIF back to submission directory
cp {project_name}.sif $PBS_O_WORKDIR/

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
        environment_activation_command=config.cluster__environment_activation_command,
        **additional_template_kwargs,
    )


def wrap_lines_with_qsub_script(
    lines: str,
) -> str:
    return QSUB_SCRIPT_HEADER + QSUB_ENV_SETUP + lines + QSUB_COMPLETION
