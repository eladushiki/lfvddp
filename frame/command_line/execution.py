import shlex
from typing import Optional

from frame.cluster.cluster_config import ClusterConfig


QSUB_SCRIPT_HEADER = """#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N {job_name}
#$ -q {queue}
#$ -l walltime={walltime}
#$ -l mem={memory}g
{gpu_line}{array_job_line}
"""

QSUB_ENV_SETUP = """
# Environment setup
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $JOB_ID"
echo "Current directory: $(pwd)"
{task_id_line}{environment_activation_command}

set -x
"""

QSUB_COMPLETION = """
# Job completion
echo "Job completed at: $(date)"
exit $?
"""

SINGULARITY_EXECUTION_LINES = """
# Main command execution
echo "Executing command on Singularity: {safe_command}"
{singularity_executable} exec lfvnn.sif {safe_command}
"""


def format_qsub_execution_script(
        config: ClusterConfig,
        command: str,
        array_jobs: Optional[int] = None,
    ) -> str:
    # Handle GPU line
    gpu_line = ""
    if config.cluster__qsub_ngpus_for_train:
        gpu_line = f"#$ -l ngpus={config.cluster__qsub_ngpus_for_train}\n"

    # Sanitize the command to prevent expansion problems
    safe_command = shlex.quote(command)
    
    return format_qsub_script(
        config=config,
        core_script_lines=SINGULARITY_EXECUTION_LINES,
        array_jobs=array_jobs,
        gpu_line=gpu_line,
        singularity_executable=config.cluster__singularity_executable,
        safe_command=safe_command,
    )


SINGULARITY_BUILD_LINES = """
# Build Singularity container with custom repository and branch
echo "Building Singularity container..."

git clone --depth 1 --filter=blob:none --sparse --branch {git_branch} {repo_url}
cd {repo_name}
git sparse-checkout set lfvddp.def

sed -e "s|^REPO_URL=.*|REPO_URL=\"{repo_url}\"|" -e "s|^BRANCH=.*|BRANCH=\"{git_branch}\"|" lfvddp.def > lfvddp-edit.def

# Build from the customized definition file
{singularity_executable} build --remote lfvddp.sif lfvddp-edit.def

# Cleanup
rm -rf lfvddp-edit.def
"""


def format_qsub_build_script(
    config: ClusterConfig,
    git_branch: str,
) -> str:
    return format_qsub_script(
        config=config,
        core_script_lines=SINGULARITY_BUILD_LINES,
        array_jobs=0,
        gpu_line="",
        git_branch=git_branch,
        repo_url=config.cluster__repo_url,
        repo_name=config.repo_name,
        singularity_executable=config.cluster__singularity_executable,
    )


def format_qsub_script(
    config: ClusterConfig,
    core_script_lines: str,
    array_jobs: Optional[int] = None,
    **additional_template_kwargs,
) -> str:
    script = wrap_lines_with_qsub_script(core_script_lines)
    
    # Handle array jobs
    array_job_line = ""
    task_id_line = ""
    if array_jobs and array_jobs > 1:
        array_job_line = f"#$ -t 1-{array_jobs}\n"
        task_id_line = 'echo "Task ID: $SGE_TASK_ID"'
    
    return script.format(
        job_name=config.cluster__qsub_job_name,
        queue=config.cluster__qsub_queue,
        walltime=config.cluster__qsub_walltime,
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
