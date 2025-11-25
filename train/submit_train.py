from pathlib import Path
from sys import argv

from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import TRAIN_DIR, path_as_in_container
from frame.submit import submit_cluster_job
from frame.cluster.cluster_config import ClusterConfig
from train.train_config import TrainConfig

CONTAINER_SINGLE_TRAIN_PATH = path_as_in_container(TRAIN_DIR / "single_train.py")


@context_controlled_execution
def submit_train(context: ExecutionContext) -> None:
    """
    Build the singularity command that runs single_train.py with current args
    """
    # Validate that we have both TrainConfig and ClusterConfig
    if not isinstance(context.config, TrainConfig):
        raise ValueError(f"Expected TrainConfig, got {context.config.__class__.__name__}")
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")
    
    # Remove the script name from argv and reconstruct the arguments
    current_args = argv[1:]
    
    # Construct the python command to run single_train.py
    python_cmd = f"python {CONTAINER_SINGLE_TRAIN_PATH}"
    
    # Add all the current arguments
    for arg in current_args:
        python_cmd += f" {arg}"
    
    # Submit the job to the cluster
    result = submit_cluster_job(
        context=context,
        command=python_cmd,
        number_of_jobs=context.config.cluster__qsub_n_jobs,
    )


if __name__ == "__main__":
    submit_train()
