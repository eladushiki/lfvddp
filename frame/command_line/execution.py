from subprocess import PIPE, Popen, check_output
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
    return check_output(wsl_command, shell=True, text=True)


def build_qsub_command(
        submitted_command: str,
        walltime: str,
        io: int,
        mem: int,
        cores: int,
        wait_job_ids: List[str],
        environment_variables: Optional[Dict[str, str]] = None,
        output_file: Optional[str] = None,
    ) -> str:
    command = f"/opt/pbs/bin/qsub -l walltime={walltime},io={io}" \
        + (f",mem={mem}g" if mem is not None else "") \
        + (f",ppn={cores}" if cores is not None else "")
    
    # if wait_job_ids:  # todo: uncomment when this become synchronous
    #     command += " -W depend=afterok"
    #     for id in wait_job_ids:
    #         command += f":{id}.pbs"

    if environment_variables:
        command += " -v "
        command += ",".join([f"{key}={value}" for key, value in environment_variables.items()])

    if output_file:
        command += f" -o {output_file}"
        
    command += f" {submitted_command}"
    
    return command


def build_qstat_command(user: str, n_jobs: int):
    return f"/opt/pbs/bin/qstat -u {user} | tail -n {n_jobs} | sed -e 's/\..*$//'"
