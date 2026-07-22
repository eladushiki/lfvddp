from os import mkdir
from pathlib import Path
from shutil import copy2

from frame.cluster.cluster_config import ClusterConfig
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import (
    CREATE_PLOTS_SCRIPT_RELATIVE,
    SINGLE_TRAIN_SCRIPT_RELATIVE,
    get_relpath_from_local_root,
    path_as_in_container,
)
from frame.submit import submit_command, submit_container_build
from train.train_config import TrainConfig


def _arguments_for_plotting_training_outputs(args: list[str]) -> list[str]:
    """Target the output directory mounted for this submitted training batch."""
    if "--plot-in-place" in args:
        return args
    return [*args, "--plot-in-place"]


@context_controlled_execution
def submit_process(context: ExecutionContext) -> None:
    """
    Build the singularity commands that run training and plotting with current args.
    """
    # Validate that we have both TrainConfig and ClusterConfig
    if not isinstance(context.config, TrainConfig):
        raise ValueError(f"Expected TrainConfig, got {context.config.__class__.__name__}")
    if not isinstance(context.config, ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")

    selected_walltime = context.prepare_next_qsub_walltime_chunk()
    uses_continuation = context.config.cluster__qsub_needs_continuation

    config_path_mapping = {}
    if not context.is_continue:
        staged_configs_directory = context.unique_out_dir / "configs"
        mkdir(staged_configs_directory)

        for config_path in context.config_paths:
            copy2(config_path, staged_configs_directory / config_path.name)
            bound_dest_path = (
                Path(context.config.config__out_dir) / "configs" / config_path.name
            )
            config_path_mapping[str(config_path)] = str(
                path_as_in_container(bound_dest_path.absolute())
            )

    # Build (or re-build) a container (if needed). Continuations reuse the existing build.
    if not context.is_continue and not context.is_no_build and not context.is_only_train:
        build_job_id = submit_container_build(context=context)
    else:
        build_job_id = None

    if context.is_continue:
        container_continue_from = str(
            get_relpath_from_local_root(
                Path(context.config.config__out_dir).absolute()
            )
        )
        current_args = ["--continue", container_continue_from]
    else:
        current_args = context.command_line_args[1:]

    # Construct the python commands to run inside the container
    train_cmd = f"python {SINGLE_TRAIN_SCRIPT_RELATIVE}"
    plot_cmd = f"python {CREATE_PLOTS_SCRIPT_RELATIVE}"

    updated_args = [config_path_mapping.get(arg, arg) for arg in current_args]

    # Add the training arguments unchanged.
    for arg in updated_args:
        train_cmd += f" {arg}"

    # Submitted plots always aggregate the outputs of this submitted batch.
    for arg in _arguments_for_plotting_training_outputs(updated_args):
        plot_cmd += f" {arg}"

    # Submit the job to the cluster
    train_jobid = submit_command(
        context=context,
        command=train_cmd,
        command_name="train",
        number_of_jobs=context.config.cluster__qsub_n_jobs,
        dependent_on_jobid=build_job_id,
    )

    if uses_continuation:
        context.record_qsub_submission(
            selected_walltime,
            train_jobid,
            context.unique_out_dir,
        )
        context.save_self_to_out_file()
        return

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
