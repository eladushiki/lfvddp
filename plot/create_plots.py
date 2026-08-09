from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from typing import Optional

from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import CONFIGS_DIR_NAME
from plot.plot_factory import PlotFactory
from plot.plotting_config import PlottingConfig


def _plot_options_from_args(
    command_line_args: Optional[list[str]] = None,
) -> tuple[Path, list[Path], bool]:
    parser = ArgumentParser()
    parser.add_argument(
        "submission_directory",
        type=Path,
        help="Submitted training directory containing configs and individual runs",
    )
    parser.add_argument(
        "additional_config_paths",
        type=Path,
        nargs="*",
        metavar="CONFIG",
        help=(
            "Additional JSON/YAML config files or directories, merged after "
            "the submission configs in the supplied order"
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run without requiring a clean Git commit",
    )
    args = parser.parse_args(command_line_args)
    submission_directory = args.submission_directory
    if not submission_directory.is_dir():
        parser.error(f"Submission directory does not exist: {submission_directory}")
    if not (submission_directory / CONFIGS_DIR_NAME).is_dir():
        parser.error(
            f"Submission directory does not contain a configs directory: "
            f"{submission_directory}"
        )
    return submission_directory, args.additional_config_paths, args.debug


def _context_arguments_for_submission_directory(
    submission_directory: Path,
    additional_config_paths: Optional[list[Path]] = None,
    debug: bool = False,
) -> list[str]:
    """Translate the plotting CLI into the existing context-controlled inputs."""
    context_arguments = [
        "--configs",
        str(submission_directory / CONFIGS_DIR_NAME),
        *(str(path) for path in additional_config_paths or []),
        "--out-dir",
        str(submission_directory),
    ]
    if debug:
        context_arguments.append("--debug")
    return context_arguments


@context_controlled_execution
def create_plots(context: ExecutionContext):
    # Make sure we have a plot config
    if not isinstance(plotting_config := context.config, PlottingConfig):
        raise TypeError("The configuration must be a PlotConfig")

    # Draw all plots
    plot_factory = PlotFactory(context=context)
    for plot in plotting_config:
        figure = plot_factory.generate_plot(plot)

        image_filename = context.unique_out_dir / plot.plot_filename
        context.save_and_document_figure(figure, image_filename)


if __name__ == "__main__":
    submission_directory, additional_config_paths, debug = _plot_options_from_args()
    argv[1:] = _context_arguments_for_submission_directory(
        submission_directory,
        additional_config_paths=additional_config_paths,
        debug=debug,
    )
    create_plots()
