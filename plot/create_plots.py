from argparse import ArgumentParser, Namespace
from pathlib import Path
from sys import argv
from typing import Optional

from frame.command_line.handle_args import create_config_from_paths
from frame.context.execution_context import (
    ExecutionContext,
    version_controlled_execution_context,
)
from frame.file_structure import CONFIGS_DIR_NAME
from frame.file_system.textual_data import expand_config_paths
from plot.plot_factory import PlotFactory
from plot.plot_utils import utils__discover_background_only_parent_directory
from plot.plotting_config import PlotScope, PlottingConfig


def _plot_options_from_args(
    command_line_args: Optional[list[str]] = None,
) -> tuple[Path, bool, bool]:
    parser = ArgumentParser()
    parser.add_argument(
        "submission_directory",
        type=Path,
        help="Submitted training directory or recursive multi-run directory",
    )
    parser.add_argument(
        "--multi-run-plots",
        action="store_true",
        help="Create aggregate plots across recursive background and signal runs",
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
    if not args.multi_run_plots and not (
        submission_directory / CONFIGS_DIR_NAME
    ).is_dir():
        parser.error(
            f"Submission directory does not contain a configs directory: "
            f"{submission_directory}"
        )
    return (
        submission_directory,
        args.multi_run_plots,
        args.debug,
    )


def _config_paths_for_plots(
    submission_directory: Path,
    multi_run_plots: bool,
) -> list[Path]:
    """Return the configuration paths needed for the requested plot scope."""
    configured_submission_directory = (
        utils__discover_background_only_parent_directory(
            str(submission_directory)
        )
        if multi_run_plots
        else submission_directory
    )
    return expand_config_paths(
        [configured_submission_directory / CONFIGS_DIR_NAME]
    )


def create_configured_plots(
    context: ExecutionContext,
    scope: PlotScope,
    performance_directory: Optional[Path] = None,
) -> None:
    """Create and persist plots assigned to one declared plot scope."""
    if not isinstance(context.config, PlottingConfig):
        raise TypeError("The configuration must be a PlotConfig")

    plot_factory = PlotFactory(context=context)
    for plot in plot_factory.plot_instructions_for_scope(
        scope,
        performance_directory=(str(performance_directory) if performance_directory else None),
    ):
        figure = plot_factory.generate_plot(plot)
        image_filename = context.unique_out_dir / plot.plot_filename
        context.save_and_document_figure(figure, image_filename)


def create_plots(
    multi_run_plots: bool,
    submission_directory: Path,
    debug: bool,
) -> None:
    """Build a plotting context directly and create the requested plot scope."""
    config_paths = _config_paths_for_plots(
        submission_directory,
        multi_run_plots,
    )
    config = create_config_from_paths(config_paths, out_dir=str(submission_directory))
    context_options = Namespace(
        continue_from=None,
        debug=debug,
        build_container=False,
        only_train=False,
    )
    with version_controlled_execution_context(
        config,
        config_paths,
        argv,
        context_options,
    ) as context:
        scope = (
            PlotScope.MULTI_RUN if multi_run_plots else PlotScope.SINGLE_SUBMISSION
        )
        create_configured_plots(
            context,
            scope,
            performance_directory=submission_directory if multi_run_plots else None,
        )


if __name__ == "__main__":
    (
        submission_directory,
        multi_run_plots,
        debug,
    ) = _plot_options_from_args()
    create_plots(
        multi_run_plots=multi_run_plots,
        submission_directory=submission_directory,
        debug=debug,
    )
