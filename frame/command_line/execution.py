import shlex
from typing import Dict, Optional

from frame.cluster.cluster_config import ClusterConfig

QSUB_TEMPLATE_SCRIPT = """#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -N {job_name}
#$ -q {queue}
#$ -pe {parallel_environment} {num_slots}
#$ -l walltime={walltime}
#$ -l mem={memory}g
#$ -l io={io}
{gpu_line}#$ -o {output_dir}/
#$ -e {error_dir}/
{environment_vars}{array_job_line}

# Environment setup
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $JOB_ID"

{singularity_executable} build --remote --build-arg GIT_URL={git_url} --build-arg GIT_BRANCH={git_branch} lfvnn.sif lfvnn.def

{task_id_line}

# Change to project root directory
cd /app/LFVNN-symmetrized
echo "Current directory: $(pwd)"

# Main command execution
echo "Executing command on Singularity: {safe_command}"
{singularity_executable} exec lfvnn.sif {safe_command}

# Job completion
echo "Job completed at: $(date)"
exit $?
"""


def format_qsub_script(
        config: ClusterConfig,
        command: str,
        environment_variables: Optional[Dict[str, str]] = None,
        array_jobs: Optional[int] = None,
        parallel_environment: str = "smp",
        num_slots: int = 1,
        output_dir: str = "/tmp",
        git_branch: str = "main",
    ) -> str:
    """
    Format the qsub script template with the provided parameters.
    
    Args:
        config: ClusterConfig object containing cluster settings
        command: The main command to execute
        environment_variables: Dictionary of environment variables to set
        array_jobs: Number of array jobs (if None, no array job)
        parallel_environment: Parallel environment specification
        num_slots: Number of slots to request
        output_dir: Directory for output and error logs
        git_branch: Git branch to use for building the container
        
    Returns:
        Formatted qsub script as a string
    """
    # Handle GPU line
    gpu_line = ""
    if config.cluster__qsub_ngpus_for_train:
        gpu_line = f"#$ -l ngpus={config.cluster__qsub_ngpus_for_train}\n"
    
    # Handle environment variables
    env_vars_line = ""
    if environment_variables:
        env_vars = ",".join([f"{key}={value}" for key, value in environment_variables.items()])
        env_vars_line = f"#$ -v {env_vars}\n"
    
    # Handle array jobs
    array_job_line = ""
    task_id_line = ""
    if array_jobs and array_jobs > 1:
        array_job_line = f"#$ -t 1-{array_jobs}\n"
        task_id_line = 'echo "Task ID: $SGE_TASK_ID"'
    
    # Format output and error directories
    error_dir = output_dir  # SGE uses same directory for both by default with -j y
    
    # Sanitize the command to prevent expansion problems
    safe_command = shlex.quote(command)
    
    return QSUB_TEMPLATE_SCRIPT.format(
        job_name=config.cluster__qsub_job_name,
        queue=config.cluster__qsub_queue or "all.q",
        parallel_environment=parallel_environment,
        num_slots=num_slots,
        walltime=config.cluster__qsub_walltime,
        memory=config.cluster__qsub_mem or 4,
        io=config.cluster__qsub_io,
        gpu_line=gpu_line,
        output_dir=output_dir,
        error_dir=error_dir,
        environment_vars=env_vars_line,
        array_job_line=array_job_line,
        git_url=config.cluster__repo_url,
        git_branch=git_branch,
        singularity_executable=config.cluster__singularity_executable,
        task_id_line=task_id_line,
        safe_command=safe_command,
    )