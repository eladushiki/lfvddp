from os import mkdir
from shutil import copy2
from sys import argv

from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import PLOT_DIR, TRAIN_DIR, path_as_in_container
from frame.submit import submit_command, submit_container_build
from frame.cluster.cluster_config import ClusterConfig
from train.train_config import TrainConfig

CONTAINER_SINGLE_TRAIN_PATH = path_as_in_container(TRAIN_DIR / "single_train.py")
CONTAINER_CREATE_PLOTS_PATH = path_as_in_container(PLOT_DIR / "create_plots.py")


@context_controlled_execution
def submit_process(context: ExecutionContext) -> None:
    """
    Build the singularity command that runs single_train.py with current args
    """
    # Validate that we have both TrainConfig and ClusterConfig
    if not isinstance(context.config, TrainConfig):
        raise ValueError(f"Expected TrainConfig, got {context.config.__class__.__name__}")
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")
    
    # Copy configuration files to destination folder
    mkdir(context.unique_out_dir / "configs")
    config_path_mapping = {}  # Map old paths to new paths
    for config_path in context.config_paths:
        dest_path = context.unique_out_dir / "configs" / config_path.name
        copy2(config_path, dest_path)
        config_path_mapping[str(config_path)] = str(dest_path)

    # Step 1: Build (or re-build) a container (if needed)
    if not context.is_no_build and not context.is_only_train:
        build_job_id = submit_container_build(context=context)
    else:
        build_job_id = None

    # Remove the script name from argv and reconstruct the arguments
    current_args = argv[1:]
    
    # Construct the python command to run single_train.py
    train_cmd = f"python {CONTAINER_SINGLE_TRAIN_PATH}"
    plot_cmd = f"python {CONTAINER_CREATE_PLOTS_PATH}"
    
    updated_args = [config_path_mapping.get(arg, arg) for arg in current_args]

    # Add all the updated arguments
    for arg in updated_args:
        train_cmd += f" {arg}"
        plot_cmd += f" {arg}"
    
    # Submit the job to the cluster
    train_jobid = submit_command(
        context=context,
        command=train_cmd,
        command_name="train",
        number_of_jobs=context.config.cluster__qsub_n_jobs,
        dependent_on_jobid=build_job_id,
    )

    if context.is_only_train:
        return
    
    plot_jobid = submit_command(
        context=context,
        command=plot_cmd,
        command_name="plot",
        number_of_jobs=1,
        dependent_on_jobid=train_jobid,
        use_gpu_if_needed=False,
    )


if __name__ == "__main__":
    submit_process()
