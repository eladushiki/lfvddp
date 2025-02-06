from logging import error
from pathlib import PurePosixPath
from subprocess import CalledProcessError, check_output
from typing import Dict, Optional
from frame.command_line.execution import build_qsub_command
from train.train_config import ClusterConfig


def submit_cluster_job(
        config: ClusterConfig,
        command: str,
        environment_variables: Optional[Dict[str, str]] = None,
        output_file: Optional[PurePosixPath] = None,
        max_tries: int = 50,
    ):
    
    # build submission command
    qsub_command = build_qsub_command(
        config=config,
        submitted_command=command,
        environment_variables=environment_variables,
        output_file=str(output_file),
    )

    for round in range(max_tries):
        try:
            _ = check_output(qsub_command)
        except CalledProcessError as e:
            stdout = e.output.decode('utf-8')
            stderr = str(e.returncode)
            error(f"Submission attempt {round} returned errorcode {stderr} with output:\n{stdout}")            

    raise RuntimeError(f"Submission failed after {max_tries} attempts")

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
