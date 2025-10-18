from logging import error
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from frame.command_line.execution import format_qsub_build_script, format_qsub_execution_script
from frame.context.execution_context import ExecutionContext
from frame.cluster.cluster_config import ClusterConfig


def submit_cluster_job(
        context: ExecutionContext,
        command: str,
        max_tries: int = 3,
        number_of_jobs: int = 1,
    ):
    
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    # Perform container build
    qsub_build_script = format_qsub_build_script(
        config=context.config,
    )

    # Save build script using ExecutionContext's save_and_document function
    build_script_filename = context.unique_out_dir / f"{context.config.cluster__qsub_job_name}_build.sh"
    stamped_build_script_filename = context.save_and_document_text(qsub_build_script, build_script_filename)
    
    qsub_a_script(context, stamped_build_script_filename, max_tries=max_tries)
    
    # Create qsub script from template
    qsub_script_content = format_qsub_execution_script(
        config=context.config,
        command=command,
        array_jobs=number_of_jobs if number_of_jobs > 1 else None,
    )
    
    # Save script using ExecutionContext's save_and_document function
    script_filename = context.unique_out_dir / f"{context.config.cluster__qsub_job_name}_submit.sh"
    stamped_script_filename = context.save_and_document_text(qsub_script_content, script_filename)

    qsub_a_script(context, stamped_script_filename, max_tries=max_tries)


def qsub_a_script(
    context: ExecutionContext,
    stamped_script_filename: Path,
    max_tries: int = 3,
):
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    # Build qsub command to submit the script.
    # ATLAS cluster can't wait to io specification in script parameters on the qsub
    qsub_command = f"qsub "\
        f"-l io={context.config.cluster__qsub_io} "\
        f"-j oe "\
        f"-o {context.unique_out_dir} "\
        f"{stamped_script_filename}"

    for round in range(max_tries):
        try:
            return check_output(qsub_command, stderr=STDOUT, shell=True)
        except CalledProcessError as e:
            stdout = e.output.decode('utf-8')
            stderr = str(e.returncode)
            error(f"Submission attempt {round} returned errorcode {stderr} with output:\n{stdout}")            
        except Exception as e:
            error(f"Unknown error with args: {e.args}\n occurred during submission attempt {round}")

    raise RuntimeError(f"Submission failed after {max_tries} attempts")
