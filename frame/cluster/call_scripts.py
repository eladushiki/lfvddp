from pathlib import Path, PurePath, PurePosixPath
from typing import Dict, List, Optional
from frame.config_handle import ExecutionContext
from frame.file_structure import RESULTS_DIR, get_remote_equivalent_path
from frame.submit import export_config_to_remote, retrieve_output_from_remote_file, submit_cluster_job
from train.train_config import ClusterConfig

RUN_PYTHON_JOB_SH_ABS_PATH = Path(__file__).parent.absolute() / "run_python_job.sh"


@export_config_to_remote
@retrieve_output_from_remote_file
def run_remote_python(
        context: ExecutionContext,
        python_script_relpath_from_workdir: PurePath,
        environment_variables: Dict[str, str] = {},
        script_arguments: List[str] = [],
        max_tries: int = 50,
        cluster_output_file: Optional[PurePosixPath] = None,
    ) -> Optional[PurePosixPath]:
    if not isinstance(config := context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {config.__class__.__name__}")

    run_python_job_sh_path = get_remote_equivalent_path(config.cluster__remote_repository_dir, RUN_PYTHON_JOB_SH_ABS_PATH)
    if not cluster_output_file:
        output_filename = context.unique_descriptor + ".out"
        cluster_output_file = get_remote_equivalent_path(config.cluster__remote_repository_dir, RESULTS_DIR / output_filename)

    environment_variables["WORKDIR"] = str(config.cluster__working_dir)
    environment_variables["SCRIPT_RELPATH"] = str(python_script_relpath_from_workdir)
    environment_variables["PYTHON_ARGS"] = f"\'{' '.join(script_arguments)}\'"

    submit_cluster_job(
        config,
        str(run_python_job_sh_path),
        environment_variables,
        cluster_output_file,
        max_tries,
    )
    # todo: poll for job finish
    return cluster_output_file
