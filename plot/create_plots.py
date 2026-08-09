from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from typing import Optional

from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import CONFIGS_DIR_NAME
from plot.plot_factory import PlotFactory
from plot.plotting_config import PlottingConfig


def _submission_directory_from_args(
    command_line_args: Optional[list[str]] = None,
) -> Path:
    parser = ArgumentParser()
    parser.add_argument(
        "submission_directory",
        type=Path,
        help="Submitted training directory containing configs and individual runs",
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
    return submission_directory


def _context_arguments_for_submission_directory(
    submission_directory: Path,
) -> list[str]:
    """Translate the plotting CLI into the existing context-controlled inputs."""
    return [
        "--configs",
        str(submission_directory / CONFIGS_DIR_NAME),
        "--out-dir",
        str(submission_directory),
    ]


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
    submission_directory = _submission_directory_from_args()
    argv[1:] = _context_arguments_for_submission_directory(submission_directory)
    create_plots()
