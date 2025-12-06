from logging import error
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Optional
from frame.command_line.execution import format_qsub_build_script, format_qsub_execution_script
from frame.context.execution_context import ExecutionContext
from frame.cluster.cluster_config import ClusterConfig
from frame.file_structure import SINGULARITY_DEFINITION_FILE
from frame.git_tools import current_git_branch, default_git_branch, get_remote_commit_hash


def submit_container_build(
        context: ExecutionContext,
        max_tries: int = 3,
) -> str:
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    build_job_name = f"{context.config.cluster__qsub_job_name}_build"

    # Perform container build
    git_branch = default_git_branch() if not context.is_debug_mode else current_git_branch()
    qsub_build_script = format_qsub_build_script(
        config=context.config,
        git_branch=git_branch,
        git_commit_hash=get_remote_commit_hash(git_branch),
    )

    # Save build script using ExecutionContext's save_and_document function
    build_script_filename = context.unique_out_dir / f"{build_job_name}.sh"
    stamped_build_script_filename = context.save_and_document_text(qsub_build_script, build_script_filename)

    # Submit build script and capture job ID
    # Pass file paths as environment variables for the script to copy
    build_env_vars = {
        "LFVDDP_DEF_PATH": str(SINGULARITY_DEFINITION_FILE.absolute()),
    }
    build_job_id = qsub_a_script(
        context=context,
        stamped_script_filename=stamped_build_script_filename,
        job_name=build_job_name,
        max_tries=max_tries,
        env_vars=build_env_vars,
    )

    return build_job_id.split('.')[0]


def submit_command(
        context: ExecutionContext,
        command: str,
        max_tries: int = 3,
        number_of_jobs: int = 1,
        dependent_on_jobid: Optional[str] = None,
    ) -> str:
    
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    exec_job_name = f"{context.config.cluster__qsub_job_name}_exec"
        
    # Create qsub script from template
    qsub_script_content = format_qsub_execution_script(
        context=context,
        command=command,
        array_jobs=number_of_jobs if number_of_jobs > 1 else None,
    )
    
    # Save script using ExecutionContext's save_and_document function
    script_filename = context.unique_out_dir / f"{exec_job_name}.sh"
    stamped_script_filename = context.save_and_document_text(qsub_script_content, script_filename)

    # Submit execution script with dependency on build job
    raw_jobid = qsub_a_script(
        context=context,
        stamped_script_filename=stamped_script_filename,
        job_name=exec_job_name,
        max_tries=max_tries,
        depends_on_success_of_jobid=dependent_on_jobid,
    )

    # Return only the numeric part of the job ID
    return raw_jobid.split('.')[0]


def qsub_a_script(
    context: ExecutionContext,
    stamped_script_filename: Path,
    job_name: str,
    max_tries: int = 3,
    depends_on_success_of_jobid: Optional[str] = None,
    env_vars: Optional[dict] = None,
) -> str:
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    # Build qsub command to submit the script.
    # ATLAS cluster can't wait to io specification in script parameters on the qsub
    qsub_command = f"qsub "\
        f"-N {job_name} "\
        f"-l io={context.config.cluster__qsub_io} "\
        f"-j oe "\
        f"-o {context.unique_out_dir} "
    
    # Add PBS dependency if specified (only run if predecessor succeeds)
    if depends_on_success_of_jobid:
        qsub_command += f"-W depend=afterok:{depends_on_success_of_jobid} "
    
    # Add environment variables if specified
    if env_vars:
        var_list = ",".join([f"{k}={v}" for k, v in env_vars.items()])
        qsub_command += f"-v {var_list} "
    
    qsub_command += f"{stamped_script_filename}"

    for round in range(max_tries):
        try:
            output = check_output(qsub_command, stderr=STDOUT, shell=True)
            output_str = output.decode('utf-8').strip()
            # PBS returns the full job ID (e.g., "12345.pbs.server.domain")
            # Return the full ID for proper dependency handling
            return output_str
        
        except CalledProcessError as e:
            stdout = e.output.decode('utf-8')
            stderr = str(e.returncode)
            error(f"Submission attempt {round} returned errorcode {stderr} with output:\n{stdout}")

        except Exception as e:
            error(f"Unknown error with args: {e.args}\n occurred during submission attempt {round}")

    raise RuntimeError(f"Submission failed after {max_tries} attempts")
