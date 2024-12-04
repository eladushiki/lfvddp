from os.path import relpath
from pathlib import Path
from typing import List, Optional
from frame.file_structure import PROJECT_ROOT
from frame.submit import output_from_remote_file, submit_cluster_job
from train.train_config import ClusterConfig

RUN_PYTHON_JOB_SH_PATH = Path(__file__).parent / "run_python_job.sh"
RUN_PYTHON_JOB_SH_RELPATH_FROM_ROOT = Path(relpath(PROJECT_ROOT, RUN_PYTHON_JOB_SH_PATH))

@output_from_remote_file
def run_remote_python(
        config: ClusterConfig,
        remote_working_dir: Path,
        python_script: Path,
        script_arguments: List[str],
        max_tries: int = 50,
        remote_output_file: Optional[Path] = None,
    ) -> Optional[Path]:
    run_python_job_sh_path = config.cluster__remote_repository_dir / RUN_PYTHON_JOB_SH_RELPATH_FROM_ROOT
    command = f"{run_python_job_sh_path} {remote_working_dir} {python_script} {' '.join(script_arguments)}"

    if remote_output_file:
        command += f" > {remote_output_file}"
    else:
        command += f" > /dev/null"
    
    submit_cluster_job(config, command, max_tries)
    return remote_output_file
