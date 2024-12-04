from contextlib import contextmanager
from functools import wraps
from logging import info, warning
from pathlib import Path
from typing import Optional
from frame.command_line.execution import build_qsub_command, execute_in_process, build_qstat_command
from frame.file_structure import PROJECT_ROOT
from frame.ssh_tools import copy_remote_file
from train.train_config import ClusterConfig


def submit_cluster_job(config: ClusterConfig, command: str, max_tries: int = 50):
    # wait for existing jobs to finish
    qstat_command = build_qstat_command(config.user, config.cluster__qstat_n_jobs)
    wait_job_ids = execute_in_process(qstat_command)[0][2:-1].split()

    # build submission command
    qsub_command = build_qsub_command(
        command,
        config.cluster__qsub_walltime,
        config.cluster__qsub_io,
        config.cluster__qsub_mem,
        config.cluster__qsub_cores,
        wait_job_ids,
    )

    for round in range(max_tries):
        out, err, return_code = execute_in_process(qsub_command)
        if return_code == 228:
            raise RuntimeError(f"Submission attempt {round}: Too many jobs submitted")
        elif return_code != 0:
            warning(f"Submission attempt {round}: received return code {return_code}, retrying")
        elif return_code == 0:
            accepted_job_id = out.split('.')[0].rstrip()
            info(f"Submitted job with id: {accepted_job_id}")
            return accepted_job_id
        
    raise RuntimeError(f"Submission failed after {max_tries} attempts")


def output_from_remote_file(submission_function):
    """
    Retrieve the output file from the remote server
    Return its context as the return value of the wrapped function.
    """
    @wraps(submission_function)
    def retrieving_function(config: ClusterConfig, *args, **kwargs) -> Optional[str]:
        remote_output_path = submission_function(config, *args, **kwargs)
        if remote_output_path:
            return retrieve_file(config, remote_output_path)
    
    return retrieving_function


def retrieve_file(config: ClusterConfig, relative_file_path_from_root: Path):
    """
    Copy a file from a remote path, within the repository,
    to the same relative path on the local machine.
    """
    remote_file_path = config.cluster__remote_repository_dir / relative_file_path_from_root
    local_file_path = PROJECT_ROOT / relative_file_path_from_root
    local_file = copy_remote_file(
        config.cluster__host_address,
        config.cluster__user,
        config.cluster__password,
        remote_file_path,
        local_file_path
    )
    if local_file:
        with open(local_file, "r") as f:
            return f.read()
    else:
        raise FileNotFoundError(f"File not found: {remote_file_path}")


# todo: revise
def prepare_submit_file(fsubname,setupLines,cmdLines,setupATLAS=True,queue="N",shortname=""):
    jobname=shortname if shortname else fsubname.rsplit('/',1)[1].split('.')[0]
    flogname=fsubname.replace('.sh','.log')
    fsub=open(fsubname,"w")
    lines=[
        "#!/bin/zsh",
        "",
        "#PBS -j oe",
        "#PBS -m n",
        "#PBS -o %s"%flogname,
        "#PBS -q %s"%queue,
        "#PBS -N %s"%jobname,
        "",
        "echo \"Starting on `hostname`, `date`\"",
        "echo \"jobs id: ${PBS_JOBID}\"",
        ""]
    if setupATLAS:
        lines+=[
            "export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase",
            "source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh",""]
    lines+=setupLines
    lines+=["","#-------------------------------------------------------------------#"]
    lines+=cmdLines
    lines+=["#-------------------------------------------------------------------#",""]
    lines+=["echo \"Done, `date`\""]
    for l in lines:
        fsub.write(l+"\n")
    fsub.close()
