from argparse import ArgumentParser, ArgumentTypeError, Namespace
from functools import wraps
from logging import warning
from pathlib import Path
from sys import argv
from typing import Callable, Optional

from frame.cluster.walltime import parse_walltime
from frame.context.execution_context import (
    create_config_from_paramters,
    version_controlled_execution_context,
)
from frame.file_structure import CONFIG_FILE_EXTENSIONS
from frame.file_system.textual_data import load_config_params_from_paths


def _walltime_argument(value: str) -> str:
    try:
        parse_walltime(value)
    except (TypeError, ValueError) as error:
        raise ArgumentTypeError(str(error)) from error
    return value


def _expand_config_paths(config_paths: list[Path]) -> list[Path]:
    """Replace config directories with their recursively discovered files."""
    expanded_paths = []
    for config_path in config_paths:
        if config_path.is_dir():
            expanded_paths.extend(
                path
                for path in sorted(config_path.rglob("*"))
                if path.is_file()
                and path.suffix.lower().removeprefix(".")
                in CONFIG_FILE_EXTENSIONS
            )
        else:
            expanded_paths.append(config_path)
    return expanded_paths


def parse_config_from_args(
    command_line_args: Optional[list[str]] = None,
) -> tuple[Optional[list[Path]], Namespace]:
    """
    A wrapper for any entry point in the project, to ensure context control.
    """
    parser = ArgumentParser()
    configuration_mode = parser.add_mutually_exclusive_group(required=True)

    # Configuration files are intentionally untyped. They are merged in the
    # order supplied, with later values overriding earlier ones.
    configuration_mode.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        dest="config_paths",
        help=(
            "Ordered configuration files or directories (JSON/YAML); directories "
            "are searched recursively and later files override earlier files"
        ),
    )
    configuration_mode.add_argument(
        "--continue",
        type=Path,
        dest="continue_from",
        metavar="LOCATION",
        help="Continue using only the context saved below LOCATION",
    )
    parser.add_argument(
        "--extra-time",
        type=_walltime_argument,
        dest="extra_time",
        metavar="HH:MM:SS",
        help="Add walltime to a saved submission before continuing it",
    )

    ## Running options
    parser.add_argument(
        "--debug", action="store_true",
        help="Run in debug mode. NOTE: Does not verify running on strict commits"
    )
    parser.add_argument(
        "--build-container",
        action="store_true",
        help="Build the container before running.",
        dest="build_container",
    )
    parser.add_argument(
        "--only-train", action="store_true",
        help="Only run the training step, skipping building and plotting steps.", dest="only_train"
    )
    parser.add_argument(
        "--out-dir", type=str,
        help="Output directory for results. Overrides one in config file. Useful for aggregating batch jobs", dest="out_dir"
    )
    parser.add_argument(
        "--plot-in-place", action="store_true",
        help="Should create plots in the output directory? Else, in a dedicated one", dest="plot_in_place"
    )
    parsed_args = argv[1:] if command_line_args is None else command_line_args
    args, unknown = parser.parse_known_args(parsed_args)

    disallowed_continue_options_used = (
        args.build_container
        or args.only_train
        or args.out_dir is not None
        or args.plot_in_place
    )
    if args.continue_from is not None and (
        unknown or disallowed_continue_options_used
    ):
        parser.error(
            "--continue LOCATION may only be combined with --extra-time and --debug"
        )
    if args.continue_from is None and args.extra_time is not None:
        parser.error("--extra-time requires --continue LOCATION")

    # Keep accepting notebook-injected arguments for fresh runs.
    if unknown:
        warning(f"Running with unknown arguments: {unknown}")

    if args.config_paths is not None:
        args.config_paths = _expand_config_paths(args.config_paths)

    return args.config_paths, args


def create_config_from_paths(
        config_paths: list[Path],
        out_dir: Optional[str] = None,
        plot_in_place: bool = False,
    ):
    return create_config_from_paramters(
        load_config_params_from_paths(config_paths),
        out_dir=out_dir,
        plot_in_place=plot_in_place,
    )


def context_controlled_execution(function: Callable):# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:
    """
    A wrapper for any entry point in the project, to ensure context control.
    """
    @wraps(function)
    def context_controlled_function(*inner_args, **inner_kwargs):
        """
        Run any decorated function in this run with the documentation of the
        configuration file parsed above.
        """
        config_paths, args = parse_config_from_args()
        config = None
        if config_paths is not None:
            config = create_config_from_paths(
                config_paths,
                out_dir=args.out_dir,
                plot_in_place=args.plot_in_place,
            )
        with version_controlled_execution_context(
            config,
            config_paths,
            argv,
            args,
        ) as context:
            function(*inner_args, **inner_kwargs, context=context)

    return context_controlled_function
