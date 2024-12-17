from pathlib import Path, PurePath
from typing import Dict, List, Optional
from frame.config_handle import ExecutionContext
from frame.file_structure import get_remote_equivalent_path
from frame.submit import export_config_to_remote, retrieve_output_from_remote_file, submit_cluster_job
from train.train_config import ClusterConfig

RUN_PYTHON_JOB_SH_ABS_PATH = Path(__file__).parent.absolute() / "run_python_job.sh"


@export_config_to_remote
@retrieve_output_from_remote_file
def run_remote_python(
        context: ExecutionContext,
        python_script_relpath_from_workdir: PurePath,
        environment_vars: Dict[str, str] = {},
        script_arguments: List[str] = [],
        max_tries: int = 50,
        cluster_output_file: Optional[Path] = None,
    ) -> Optional[Path]:
    if not isinstance(config := context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {config.__class__.__name__}")

    run_python_job_sh_path = get_remote_equivalent_path(config.cluster__remote_repository_dir, RUN_PYTHON_JOB_SH_ABS_PATH)
    
    environment_vars["WORKDIR"] = str(config.cluster__working_dir)
    environment_vars["SCRIPT_RELPATH"] = str(python_script_relpath_from_workdir)
    environment_vars["PYTHON_ARGS"] = f"\'{' '.join(script_arguments)}\'"
    arguments_digest = ",".join([f"{key}={value}" for key, value in environment_vars.items()])
    command = f"-v {arguments_digest} {run_python_job_sh_path}"

    if cluster_output_file:
        command += f" > {cluster_output_file}"
    else:
        command += f" > /dev/null"
    
    submit_cluster_job(config, command, max_tries)
    return cluster_output_file
