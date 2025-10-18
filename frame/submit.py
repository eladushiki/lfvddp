from logging import error
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
import tarfile
from typing import Optional, List, Tuple
from frame.command_line.execution import format_qsub_build_script, format_qsub_execution_script
from frame.context.execution_context import ExecutionContext
from frame.cluster.cluster_config import ClusterConfig
from frame.file_structure import CONFIGS_DIR, SINGULARITY_DEFINITION_FILE, TARBALL_FILE_EXTENSION
from frame.git_tools import current_git_branch, default_git_branch


def submit_cluster_job(
        context: ExecutionContext,
        command: str,
        max_tries: int = 3,
        number_of_jobs: int = 1,
    ):
    
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    # Define job names for build and execution
    build_job_name = f"{context.config.cluster__qsub_job_name}_build"
    exec_job_name = f"{context.config.cluster__qsub_job_name}_exec"

    # Create tarball of configs directory
    configs_tarball = context.unique_out_dir / f"configs.{TARBALL_FILE_EXTENSION}"
    with tarfile.open(configs_tarball, "w:gz") as tar:
        tar.add(CONFIGS_DIR, arcname="configs")
    
    # Prepare stagein files for build job: lfvddp.def and configs.tar.gz
    stagein_files = [
        (str(SINGULARITY_DEFINITION_FILE), "lfvddp.def"),
        (str(configs_tarball), "configs.tar.gz"),
    ]

    # Perform container build
    qsub_build_script = format_qsub_build_script(
        config=context.config,
        git_branch=default_git_branch() if not context.is_debug_mode else current_git_branch()
    )

    # Save build script using ExecutionContext's save_and_document function
    build_script_filename = context.unique_out_dir / f"{build_job_name}.sh"
    stamped_build_script_filename = context.save_and_document_text(qsub_build_script, build_script_filename)
    
    # Submit build script and capture job ID
    build_job_id = qsub_a_script(
        context=context,
        stamped_script_filename=stamped_build_script_filename,
        job_name=build_job_name,
        max_tries=max_tries,
        stagein_files=stagein_files,
    )
    
    # Create qsub script from template
    qsub_script_content = format_qsub_execution_script(
        config=context.config,
        command=command,
        array_jobs=number_of_jobs if number_of_jobs > 1 else None,
    )
    
    # Save script using ExecutionContext's save_and_document function
    script_filename = context.unique_out_dir / f"{exec_job_name}.sh"
    stamped_script_filename = context.save_and_document_text(qsub_script_content, script_filename)

    # Submit execution script with dependency on build job
    qsub_a_script(
        context=context,
        stamped_script_filename=stamped_script_filename,
        job_name=exec_job_name,
        max_tries=max_tries,
        depends_on_success_of_jobid=build_job_id,
    )


def qsub_a_script(
    context: ExecutionContext,
    stamped_script_filename: Path,
    job_name: str,
    max_tries: int = 3,
    depends_on_success_of_jobid: Optional[str] = None,
    stagein_files: Optional[List[Tuple[str, str]]] = None,
):
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    # Build qsub command to submit the script.
    # ATLAS cluster can't wait to io specification in script parameters on the qsub
    qsub_command = f"qsub "\
        f"-N {job_name} "\
        f"-l io={context.config.cluster__qsub_io} "\
        f"-j oe "\
        f"-o {context.unique_out_dir} "
    
    # Add PBS/Torque dependency if specified (only run if predecessor succeeds)
    if depends_on_success_of_jobid:
        qsub_command += f"-W depend=afterok:{depends_on_success_of_jobid} "
    
    # Add PBS/Torque stagein for files if specified
    if stagein_files:
        stagein_spec = ",".join([f"{src}@{dst}" for src, dst in stagein_files])
        qsub_command += f"-W stagein={stagein_spec} "
    
    qsub_command += f"{stamped_script_filename}"

    for round in range(max_tries):
        try:
            output = check_output(qsub_command, stderr=STDOUT, shell=True)
            output_str = output.decode('utf-8').strip()
            # PBS/Torque returns just the job ID (e.g., "12345.server.domain")
            job_id = output_str.split('.')[0] if '.' in output_str else output_str
            return job_id
        
        except CalledProcessError as e:
            stdout = e.output.decode('utf-8')
            stderr = str(e.returncode)
            error(f"Submission attempt {round} returned errorcode {stderr} with output:\n{stdout}")

        except Exception as e:
            error(f"Unknown error with args: {e.args}\n occurred during submission attempt {round}")

    raise RuntimeError(f"Submission failed after {max_tries} attempts")
