from typing import Dict, Optional

from train.train_config import ClusterConfig

def build_qsub_command(
        config: ClusterConfig,
        submitted_command: str,
        environment_variables: Optional[Dict[str, str]] = None,
        number_of_jobs: int = 1,
    ) -> str:
    command = f"/opt/pbs/bin/qsub -l " \
        + (f"ngpus={config.cluster__qsub_ngpus_for_train}" if config.cluster__qsub_ngpus_for_train else "") \
        + f",io={config.cluster__qsub_io}"
        # + f"walltime={config.cluster__qsub_walltime}" \
        # + (f",mem={config.cluster__qsub_mem}g" if config.cluster__qsub_mem is not None else "") \
        # + (f",ppn={config.cluster__qsub_cores}" if config.cluster__qsub_cores is not None else "") \
    
    # todo: format this so any config would be optional and taken from config
    # and also, remove all duplicate configs from shell file
    
    if config.cluster__qsub_job_name:
        command += f" -N {config.cluster__qsub_job_name}"

    if config.cluster__qsub_queue:
        command += f" -q {config.cluster__qsub_queue}"

    if environment_variables:
        command += " -v "
        command += ",".join([f"{key}={value}" for key, value in environment_variables.items()])

    command += f" -j oe"

    if number_of_jobs > 1:
        command += f" -J 1-{number_of_jobs}"
        
    command += f" {submitted_command}"

    return command
