from subprocess import PIPE, Popen, check_output
from typing import Dict, List, Optional

from train.train_config import ClusterConfig

def execute_in_process(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        stderr=PIPE,
    shell=True)
    out,err=process.communicate()
    returncode=process.returncode
    return str(out),str(err),returncode

def build_qsub_command(
        config: ClusterConfig,
        submitted_command: str,
        environment_variables: Optional[Dict[str, str]] = None,
        output_file: Optional[str] = None,
    ) -> str:
    command = f"/opt/pbs/bin/qsub -l walltime={config.cluster__qsub_walltime},io={config.cluster__qsub_io}" \
        + (f",mem={config.cluster__qsub_mem}g" if config.cluster__qsub_mem is not None else "") \
        + (f",ppn={config.cluster__qsub_cores}" if config.cluster__qsub_cores is not None else "")
    
    if environment_variables:
        command += " -v "
        command += ",".join([f"{key}={value}" for key, value in environment_variables.items()])

    if output_file:
        command += f" -o {output_file}"
        
    command += f" {submitted_command}"
    
    return command
