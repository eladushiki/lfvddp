from logging import error
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict, Optional
from frame.command_line.execution import build_qsub_command
from train.train_config import ClusterConfig


def submit_cluster_job(
        config: ClusterConfig,
        command: str,
        environment_variables: Optional[Dict[str, str]] = None,
        max_tries: int = 3,
    ):
    
    # build submission command
    qsub_command = build_qsub_command(
        config=config,
        submitted_command=command,
        environment_variables=environment_variables,
        number_of_jobs=config.cluster__qsub_n_jobs,
    )

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
