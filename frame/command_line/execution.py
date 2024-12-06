from os import system
from subprocess import PIPE, Popen
from typing import Dict, List, Optional

def execute_in_process(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        stderr=PIPE,
    shell=True)
    out,err=process.communicate()
    returncode=process.returncode
    return str(out),str(err),returncode


def execute_in_wsl(cmd_command: str):
    """
    Used to run commands from a windows machine in a WSL shell.
    Double quotes and quote signs may cause execution problems.
    """
    if '"' in cmd_command:
        separator = "'"
    else:
        separator = '"'
    wsl_command = f"wsl ~ -e sh -c {separator}{cmd_command}{separator}"
    return system(wsl_command)


def build_qsub_command(
        submitted_command: str,
        walltime: str,
        io: int,
        mem: int,
        cores: int,
        wait_job_ids: List[str],
        command_line_arguments: Optional[List[str]] = None,
        environment_variables: Optional[Dict[str, str]] = None,
    ) -> str:
    command = f"qsub -l walltime={walltime},io={io}" \
        + (f",mem={mem}g" if mem is not None else "") \
        + (f",ppn={cores}" if cores is not None else "")
    
    if wait_job_ids:
        command += " -W depend=afterok"
        for id in wait_job_ids:
            command += f":{id}.wipp-pbs"

    if environment_variables:
        command += " -v "
        command += ",".join([f"{key}=\"{value}\"" for key, value in environment_variables.items()])
        
    command += f" {submitted_command}"
    
    if command_line_arguments:
        for argument in command_line_arguments:
            command += f" {argument}"

    return command


def build_qstat_command(user: str, n_jobs: int):
    command = f"qstat -u {user} | tail -n {n_jobs} | sed -e 's/\..*$//' | tr '\n' ' '"
    return command
