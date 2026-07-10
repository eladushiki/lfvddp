from os import mkdir
from pathlib import Path
from shutil import copy2
from sys import argv

from frame.cluster.cluster_config import ClusterConfig
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import (
    CONTAINER_CREATE_PLOTS_PATH,
    CONTAINER_SINGLE_TRAIN_PATH,
    SUBMIT_TRAIN_SCRIPT_NAME,
    path_as_in_container,
)
from frame.submit import submit_command, submit_container_build
from train.train_config import TrainConfig


def _replace_or_append_continue_from(
    args: list[str],
    container_continue_from: str,
    include_continue_flag: bool = True,
) -> list[str]:
    updated_args = []
    skip_next = False

    for arg in args:
        if skip_next:
            skip_next = False
            continue

        if arg == "--continue-from":
            updated_args.extend([arg, container_continue_from])
            skip_next = True
            continue
        if arg.startswith("--continue-from="):
            updated_args.append(f"--continue-from={container_continue_from}")
            continue

        updated_args.append(arg)

    if include_continue_flag and "--continue" not in updated_args:
        updated_args.append("--continue")
    if not any(arg == "--continue-from" or arg.startswith("--continue-from=") for arg in updated_args):
        updated_args.extend(["--continue-from", container_continue_from])

    return updated_args


def _arguments_for_plotting_training_outputs(args: list[str]) -> list[str]:
    """Target the output directory mounted for this submitted training batch."""
    if "--plot-in-place" in args:
        return args
    return [*args, "--plot-in-place"]


def _context_to_continue(context: ExecutionContext) -> ExecutionContext:
    if context.continue_from is not None:
        return ExecutionContext.load_from_run_dir(context.continue_from)

    continuation_context = ExecutionContext.find_stamped_run_context(
        Path(context.config.config__out_dir),
        context.config.config__dirsafe_runtag,
        entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
        require_continuation=True,
    )
    if continuation_context is None:
        raise RuntimeError("Could not find a stamped submit context to continue.")
    return continuation_context


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

    state_context = _context_to_continue(context) if context.is_continue else context
    selected_walltime = state_context.next_qsub_walltime_chunk()
    if selected_walltime is None:
        raise RuntimeError(f"No remaining walltime chunks to submit for {state_context.unique_out_dir}.")

    submit_context = state_context if context.is_continue else context
    submit_context.config.use_walltime_chunk(selected_walltime)
    uses_continuation = state_context.config.cluster__qsub_needs_continuation

    config_path_mapping = {}  # Map old paths to new paths
    if not context.is_continue:
        # Copy configuration files to destination folder
        mkdir(context.unique_out_dir / "configs")
        for config_path in context.config_paths:
            dest_path = context.unique_out_dir / "configs" / config_path.name
            copy2(config_path, dest_path)

            # Generate copied config paths as would appear in container
            bound_dest_path = Path(context.config.config__out_dir) / "configs" / config_path.name
            config_path_mapping[str(config_path)] = str(path_as_in_container(bound_dest_path.absolute()))
    else:
        for config_path in context.config_paths:
            bound_dest_path = Path(submit_context.config.config__out_dir) / "configs" / config_path.name
            config_path_mapping[str(config_path)] = str(path_as_in_container(bound_dest_path.absolute()))

    # Build (or re-build) a container (if needed). Continuations reuse the existing build.
    if not context.is_continue and not context.is_no_build and not context.is_only_train:
        build_job_id = submit_container_build(context=submit_context)
    else:
        build_job_id = None

    # Remove the script name from argv and reconstruct the arguments
    current_args = argv[1:]
    if uses_continuation:
        container_continue_from = str(path_as_in_container(Path(submit_context.config.config__out_dir).absolute()))
        current_args = _replace_or_append_continue_from(
            current_args,
            container_continue_from,
            include_continue_flag=context.is_continue,
        )

    # Construct the python commands to run inside the container
    train_cmd = f"python {CONTAINER_SINGLE_TRAIN_PATH}"
    plot_cmd = f"python {CONTAINER_CREATE_PLOTS_PATH}"

    updated_args = [config_path_mapping.get(arg, arg) for arg in current_args]

    # Add the training arguments unchanged.
    for arg in updated_args:
        train_cmd += f" {arg}"

    # Submitted plots always aggregate the outputs of this submitted batch.
    for arg in _arguments_for_plotting_training_outputs(updated_args):
        plot_cmd += f" {arg}"

    # Submit the job to the cluster
    train_jobid = submit_command(
        context=submit_context,
        command=train_cmd,
        command_name="train",
        number_of_jobs=submit_context.config.cluster__qsub_n_jobs,
        dependent_on_jobid=build_job_id,
    )

    if uses_continuation:
        state_context.record_qsub_submission(
            selected_walltime,
            train_jobid,
            submit_context.unique_out_dir,
        )
        state_context.save_self_to_out_file()
        return

    if context.is_only_train:
        return

    plot_jobid = submit_command(
        context=submit_context,
        command=plot_cmd,
        command_name="plot",
        number_of_jobs=1,
        dependent_on_jobid=train_jobid,
        use_gpu_if_needed=False,
    )


if __name__ == "__main__":
    submit_process()
