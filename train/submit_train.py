from pathlib import Path
from sys import argv

from frame.cluster.call_scripts import RUN_PYTHON_JOB_SH_ABS_PATH, run_remote_python
from frame.cluster.remote_version_control import is_same_version_as_remote
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import get_relpath_from_local_root
from train.aggregate_train_results import aggregate_train_results
from train.train_config import TrainConfig

SINGLE_TRAIN_RELPATH_FROM_ROOT = get_relpath_from_local_root(Path(__file__).parent.absolute() / "single_train.py")


@context_controlled_execution
def submit_train(context: ExecutionContext) -> None:
    if not isinstance(context.config, TrainConfig):
        raise ValueError(f"Expected TrainConfig, got {context.config.__class__.__name__}")
    
    # Prepare training job
    ## Verify commit hash matching with remote repository
    # is_same_version_as_remote(context)  # todo: this does not work asynchronously. Anyway, working commit are documented and running on strict commits.

    # Submit training job
    run_remote_python(
        context=context,
        run_python_bash_script_abspath=RUN_PYTHON_JOB_SH_ABS_PATH,
        workdir_at_cluster_abspath=context.config.cluster__project_root_at_cluster_abspath,
        environment_activation_script_abspath=context.config.cluster__environment_activation_script_at_cluster_abspath,
        python_script_relpath_from_workdir_at_cluster=SINGLE_TRAIN_RELPATH_FROM_ROOT,
        script_arguments=argv[1:],
        number_of_jobs=context.config.cluster__qsub_n_jobs,
    )

    # todo: this does not work asynchronously. Poll for jobs.
    # aggregate_train_results(
    #     context.unique_out_dir,
    # )

if __name__ == "__main__":
    submit_train()
