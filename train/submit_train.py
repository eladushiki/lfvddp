from os import mkdir
from pathlib import Path
from shlex import join
from shutil import copy2

from frame.cluster.cluster_config import ClusterConfig
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import (
    CONFIGS_DIR_NAME,
    CREATE_PLOTS_SCRIPT_RELATIVE,
    SINGLE_TRAIN_SCRIPT_RELATIVE,
    get_relpath_from_local_root,
    path_as_in_container,
)
from frame.submit import submit_command, submit_container_build
from train.train_config import TrainConfig


def _stage_config_files(context: ExecutionContext) -> str:
    """Copy configs and return their bound directory inside the container."""
    staged_configs_directory = context.unique_out_dir / CONFIGS_DIR_NAME
    mkdir(staged_configs_directory)

    for index, config_path in enumerate(context.config_paths):
        # The index preserves merge order and prevents equal basenames from
        # separate pack directories from overwriting one another.
        staged_name = f"{index:04d}_{config_path.name}"
        copy2(config_path, staged_configs_directory / staged_name)

    bound_configs_directory = (
        Path(context.config.config__out_dir) / CONFIGS_DIR_NAME
    )
    return str(path_as_in_container(bound_configs_directory.absolute()))


def _replace_config_arguments(
    command_line_args: list[str],
    container_configs_directory: str,
) -> list[str]:
    """Replace host config arguments with staged paths visible in the container."""
    arguments = command_line_args[1:]
    try:
        configs_index = arguments.index("--configs")
    except ValueError as error:
        raise ValueError("Fresh submissions require --configs arguments.") from error

    configs_end = configs_index + 1
    while (
        configs_end < len(arguments)
        and not arguments[configs_end].startswith("-")
    ):
        configs_end += 1

    return [
        *arguments[:configs_index],
        "--configs",
        container_configs_directory,
        *arguments[configs_end:],
    ]


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

    if not context.is_continue:
        container_configs_directory = _stage_config_files(context)

    # Build a container when explicitly requested. Continuations reuse the existing build.
    if context.is_build_container and not context.is_continue and not context.is_only_train:
        build_job_id = submit_container_build(context=context)
    else:
        build_job_id = None

    if context.is_continue:
        container_continue_from = str(
            get_relpath_from_local_root(
                Path(context.config.config__out_dir).absolute()
            )
        )
        current_args = [
            "--continue",
            container_continue_from,
            "--epochs-target",
            str(context.config.train__epochs),
        ]
    else:
        current_args = _replace_config_arguments(
            context.command_line_args,
            container_configs_directory,
        )

    # Construct the python commands to run inside the container
    train_cmd = join(["python", str(SINGLE_TRAIN_SCRIPT_RELATIVE), *current_args])

    # Submit the job to the cluster
    train_jobid = submit_command(
        context=context,
        command=train_cmd,
        command_name="train",
        number_of_jobs=context.config.cluster__qsub_n_jobs,
        dependent_on_jobid=build_job_id,
    )

    context.record_qsub_submission(
        selected_walltime,
        train_jobid,
        context.unique_out_dir,
    )
    context.save_self_to_out_file()

    if context.next_qsub_walltime_chunk() is not None:
        return

    if context.is_only_train:
        return

    container_submission_directory = path_as_in_container(
        Path(context.config.config__out_dir).absolute()
    )
    plot_cmd = join([
        "python",
        str(CREATE_PLOTS_SCRIPT_RELATIVE),
        str(container_submission_directory),
        *(["--debug"] if context.is_debug_mode else []),
    ])
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
