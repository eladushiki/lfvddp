from pathlib import Path
from typing import Optional

from frame.cluster.cluster_config import ClusterConfig
from frame.context.execution_context import ExecutionContext
from frame.file_structure import CONFIGS_DIR, CONTAINER_PROJECT_ROOT, PROJECT_NAME, path_as_in_container


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
echo "Executing command on Singularity: {command}"
{singularity_executable} exec --cleanenv --pwd {container_project_root} --bind {singularity_bindings} {sif_path} {command}
"""


def format_qsub_execution_script(
        context: ExecutionContext,
        command: str,
        array_jobs: Optional[int] = None,
    ) -> str:
    config: ClusterConfig = context.config

    # Handle GPU line
    gpu_line = ""
    if config.cluster__qsub_ngpus_for_train:
        gpu_line = f"#$ -l ngpus={config.cluster__qsub_ngpus_for_train}\n"

    singularity_bindings = ",".join([
        f"{Path(local_path).absolute()}:{container_path}"
        for local_path, container_path in context.config.config__bind_directories.items()
    ] + [f"{context.unique_out_dir.absolute()}:{path_as_in_container(Path(config.config__out_dir).absolute())}"])

    # Pass command directly without quoting for Singularity
    return format_qsub_script(
        config=config,
        core_script_lines=SINGULARITY_EXECUTION_LINES,
        array_jobs=array_jobs,
        gpu_line=gpu_line,
        singularity_executable=config.cluster__singularity_executable,
        container_project_root=CONTAINER_PROJECT_ROOT,
        singularity_bindings=singularity_bindings,
        sif_path=LOCAL_PROJECT_ROOT / f"{PROJECT_NAME}.sif",
        command=command,
    )


SINGULARITY_BUILD_LINES = """
# Build Singularity container with custom repository and branch
echo "Building Singularity container..."

# Create working directory for build
BUILD_DIR=$(mktemp -d)
cd $BUILD_DIR

# Copy definition file from source path (passed as environment variable)
echo "Copying {project_name}.def file from $LFVDDP_DEF_PATH..."
cp $LFVDDP_DEF_PATH ./{project_name}.def

# Extract configs tarball from source path
echo "Extracting configs from {configs_tarball_path}..."
tar -xzf {configs_tarball_path}

# Customize the definition file with repository URL, branch, and commit hash
# The commit hash is added as a comment to bust Singularity's layer cache
sed -e "s|REPO_URL=.*|REPO_URL=\"{repo_url}\"|" \
    -e "s|BRANCH=.*|BRANCH=\"{git_branch}\"|" \
    -e "s|CONTAINER_CONFIGS_DIR=.*|CONTAINER_CONFIGS_DIR=\"{container_configs_dir}\"|" \
    -e "s|CONTAINER_PROJECT_ROOT=.*|CONTAINER_PROJECT_ROOT=\"{container_project_root}\"|" \
    -e "s|# Cache-busting commit: PLACEHOLDER|# Cache-busting commit: {git_commit_hash}|" \
    {project_name}.def > {project_name}-edit.def

# Build from the customized definition file
echo "Building container..."
{singularity_executable} build --remote {project_name}.sif {project_name}-edit.def
# Copy the built container and configs back to submission directory
cp {project_name}.sif $PBS_O_WORKDIR/
cp -r configs $PBS_O_WORKDIR/

# Cleanup
cd $PBS_O_WORKDIR
rm -rf $BUILD_DIR
"""


def format_qsub_build_script(
    config: ClusterConfig,
    git_branch: str,
    git_commit_hash: str,
    configs_tarball_path: str,
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
        configs_tarball_path=configs_tarball_path,
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
        task_id_line = 'echo "Task ID: $SGE_TASK_ID"\n'
    
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
