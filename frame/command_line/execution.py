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
set -x
"""

QSUB_COMPLETION = """
# Job completion
JOB_EXIT_CODE=$?
echo "Job completed at: $(date)"
exit $JOB_EXIT_CODE
"""

# Singularity exec command. A few comments:
# --no-mount tmp: relieves pressure on file descriptors when running many jobs in parallel
# --cleanenv: avoids messing with host environment, i.e. python stuff

SINGULARITY_EXECUTION_LINES = r"""
# Main command execution
echo "Executing command on Singularity: {command}"

# Keep unsquashfs parallelism bounded to reduce file-descriptor pressure.
export SINGULARITY_UNSQUASHFS_PROCS=${{SINGULARITY_UNSQUASHFS_PROCS:-8}}
# Older Singularity builds may ignore *_UNSQUASHFS_PROCS but honor explicit unsquashfs options.
export SINGULARITY_UNSQUASHFS_OPTS="${{SINGULARITY_UNSQUASHFS_OPTS:--processors $SINGULARITY_UNSQUASHFS_PROCS}}"
echo "Unsquashfs config: procs=$SINGULARITY_UNSQUASHFS_PROCS opts='$SINGULARITY_UNSQUASHFS_OPTS'"
echo "Open-files soft limit: $(ulimit -n)"

# Stagger container extraction across array tasks to avoid synchronized sandbox extraction.
TASK_ID="${{PBS_ARRAY_INDEX:-}}"
if [ -z "$TASK_ID" ]; then
    TASK_ID=$(echo "$PBS_JOBID" | sed -n 's/.*\[\([0-9]*\)\].*/\1/p')
fi
if [ -n "$TASK_ID" ] && [ "$TASK_ID" -gt 1 ]; then
    STAGGER_STEP_SEC=${{SINGULARITY_ARRAY_STAGGER_STEP_SEC:-180}}
    STAGGER_JITTER_SEC=${{SINGULARITY_ARRAY_STAGGER_JITTER_SEC:-30}}
    DELAY=$(( (TASK_ID - 1) * STAGGER_STEP_SEC + RANDOM % (STAGGER_JITTER_SEC + 1) ))
    echo "Waiting $DELAY seconds before container extraction..."
    sleep "$DELAY"
fi

SINGULARITY_CMD="{singularity_executable} exec --no-mount tmp --cleanenv --pwd {container_project_root} --bind {singularity_bindings} {container_path} {command}"
MAX_ATTEMPTS=${{SINGULARITY_EXEC_MAX_ATTEMPTS:-5}}
ATTEMPT=1
EXIT_CODE=1

while [ "$ATTEMPT" -le "$MAX_ATTEMPTS" ]; do
    echo "Singularity attempt $ATTEMPT/$MAX_ATTEMPTS"
    EXEC_LOG=$(mktemp)
    set +e
    eval "$SINGULARITY_CMD" >"$EXEC_LOG" 2>&1
    EXIT_CODE=$?
    set -e
    cat "$EXEC_LOG"

    if [ "$EXIT_CODE" -eq 0 ]; then
        rm -f "$EXEC_LOG"
        break
    fi

    if grep -qi "Too many open files" "$EXEC_LOG" && [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; then
        RETRY_DELAY_BASE=${{SINGULARITY_RETRY_DELAY_BASE_SEC:-120}}
        RETRY_DELAY_JITTER=${{SINGULARITY_RETRY_DELAY_JITTER_SEC:-30}}
        RETRY_DELAY=$((ATTEMPT * RETRY_DELAY_BASE + RANDOM % (RETRY_DELAY_JITTER + 1)))
        echo "Detected open-file exhaustion, retrying in ${{RETRY_DELAY}}s..."
        rm -f "$EXEC_LOG"
        sleep "$RETRY_DELAY"
        ATTEMPT=$((ATTEMPT + 1))
        continue
    fi

    rm -f "$EXEC_LOG"
    break
done

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "Singularity command failed after $ATTEMPT attempt(s) with exit code $EXIT_CODE"
    exit "$EXIT_CODE"
fi
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
