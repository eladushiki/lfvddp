from pathlib import Path, PurePath, PurePosixPath
from typing import Dict, List, Optional
from frame.config_handle import ExecutionContext
from frame.file_structure import JOB_OUTPUT_FILE_NAME
from frame.submit import submit_cluster_job
from train.train_config import ClusterConfig

RUN_PYTHON_JOB_SH_ABS_PATH = Path(__file__).parent.absolute() / "run_python_job.sh"


def run_remote_python(
        context: ExecutionContext,
        run_python_bash_script_abspath: Path,
        workdir_at_cluster_abspath: PurePosixPath,
        environment_activation_script_abspath: PurePosixPath,
        python_script_relpath_from_workdir_at_cluster: PurePath,
        environment_variables: Dict[str, str] = {},
        script_arguments: List[str] = [],
        output_dir: Optional[Path] = None,
        output_filename: str = JOB_OUTPUT_FILE_NAME,
        max_tries: int = 3,
    ):
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    # To allow for nested submission while aggregating the outputs in nested directories, override out_dir in the submitted jobs
    script_arguments += ["--out-dir", str(context.unique_out_dir)]

    environment_variables["WORKDIR"] = str(workdir_at_cluster_abspath)
    environment_variables["OUTPUT_DIR"] = str(output_dir)
    environment_variables["OUTPUT_FILENAME"] = output_filename
    environment_variables["ENV_ACTIVATION_SCRIPT"] = str(environment_activation_script_abspath)
    environment_variables["SCRIPT_RELPATH"] = str(python_script_relpath_from_workdir_at_cluster)
    environment_variables["PYTHON_ARGS"] = f"\'{' '.join(script_arguments)}\'"

    return submit_cluster_job(
        context=context,
        command=str(run_python_bash_script_abspath),
        environment_variables=environment_variables,
        max_tries=max_tries,
    )
