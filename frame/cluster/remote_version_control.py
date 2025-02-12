from pathlib import Path
from frame.cluster.call_scripts import RUN_PYTHON_JOB_SH_ABS_PATH, run_remote_python
from frame.config_handle import ExecutionContext
from frame.git_tools import get_commit_hash
from train.train_config import ClusterConfig

IS_SAME_AS_COMMIT_PY_ABS_PATH = Path(__file__).parent.absolute() / "is_same_as_commit.py"
IS_SAME_AS_COMMIT_DEFAULT_OUTPUT_FILENAME = "is_same_as_commit_output.txt"

def is_same_version_as_remote(
        context: ExecutionContext,
    ) -> bool:
    """
    Call this from the user's machine before submitting a
    job to verify that the remote commit hash matches the
    local one.
    """
    if not isinstance((config := context.config), ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")
    
    local_commit_hash = get_commit_hash()

    comparison_result = run_remote_python(  # todo: deprecated, revise
        context=context,
        run_python_bash_script_abspath=RUN_PYTHON_JOB_SH_ABS_PATH,
        workdir_at_cluster_abspath=config.cluster__project_root_at_cluster_abspath,
        environment_activation_script_abspath=config.cluster__environment_activation_script_at_cluster_abspath,
        python_script_relpath_from_workdir_at_cluster=IS_SAME_AS_COMMIT_PY_ABS_PATH,
        script_arguments=["--commit-hash", local_commit_hash],
        output_filename=IS_SAME_AS_COMMIT_DEFAULT_OUTPUT_FILENAME,
        number_of_jobs=1,  # This job is singular
    )

    # return comparison_result == "0"  # todo: this is not correct, the output is *****.pbs job name. Need to synchronously poll the output file
