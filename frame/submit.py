from logging import error
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict, Optional
from frame.command_line.execution import format_qsub_script
from frame.context.execution_context import ExecutionContext
from frame.cluster.cluster_config import ClusterConfig


def submit_cluster_job(
        context: ExecutionContext,
        command: str,
        environment_variables: Optional[Dict[str, str]] = None,
        max_tries: int = 3,
        number_of_jobs: int = 1,
    ):
    
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    # Create qsub script from template
    qsub_script_content = format_qsub_script(
        config=context.config,
        command=command,
        environment_variables=environment_variables,
        array_jobs=number_of_jobs if number_of_jobs > 1 else None,
        output_dir=str(context.unique_out_dir),
    )
    
    # Save script using ExecutionContext's save_and_document function
    script_filename = context.unique_out_dir / f"{context.config.cluster__qsub_job_name}_submit.sh"
    context.save_and_document_text(qsub_script_content, script_filename)
    
    # Make script executable
    script_filename.chmod(0o755)
    
    # Build qsub command to submit the script
    qsub_command = f"qsub {script_filename}"

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
