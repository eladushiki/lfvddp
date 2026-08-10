from argparse import ArgumentParser
from pathlib import Path
from sys import argv
from typing import Optional

from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from plot.create_plots import create_configured_plots
from plot.plot_factory import PlotFactory


def _performance_plot_options_from_args(
    command_line_args: Optional[list[str]] = None,
) -> tuple[Path, list[Path], bool]:
    parser = ArgumentParser()
    parser.add_argument(
        "performance_directory",
        type=Path,
        help="Directory recursively containing background and signal runs",
    )
    parser.add_argument(
        "config_paths",
        type=Path,
        nargs="+",
        metavar="CONFIG",
        help="Plot configuration files or directories",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run without requiring a clean Git commit",
    )
    args = parser.parse_args(command_line_args)
    if not args.performance_directory.is_dir():
        parser.error(
            "Performance directory does not exist: "
            f"{args.performance_directory}"
        )
    return args.performance_directory, args.config_paths, args.debug


@context_controlled_execution
def create_performance_plots(
    context: ExecutionContext,
    performance_directory: Path,
) -> None:
    create_configured_plots(
        context,
        PlotFactory.MULTI_RUN_ENTRYPOINT,
        performance_directory=performance_directory,
    )


if __name__ == "__main__":
    performance_directory, config_paths, debug = _performance_plot_options_from_args()
    argv[1:] = [
        "--configs",
        *(str(path) for path in config_paths),
        "--out-dir",
        str(performance_directory),
        *( ["--debug"] if debug else [] ),
    ]
    create_performance_plots(performance_directory=performance_directory)
