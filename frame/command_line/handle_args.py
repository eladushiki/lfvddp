from logging import warning
from sys import argv
from argparse import ArgumentParser, Namespace
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from frame.context.execution_context import (
    create_config_from_paramters,
    version_controlled_execution_context,
)
from frame.file_system.textual_data import load_config_params_from_paths


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
        help="Ordered configuration files (JSON/YAML); later files override earlier files",
    )
    configuration_mode.add_argument(
        "--continue",
        type=Path,
        dest="continue_from",
        metavar="LOCATION",
        help="Continue using only the context saved below LOCATION",
    )

    ## Running options
    parser.add_argument(
        "--debug", action="store_true",
        help="Run in debug mode. NOTE: Does not verify running on strict commits"
    )
    parser.add_argument(
        "--no-build", action="store_true",
        help="Do not build the container before running. Useful for debug, prone to errors.", dest="no_build"
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

    if args.continue_from is not None and (
        unknown
        or len(parsed_args) not in (2, 3)
        or parsed_args[0] != "--continue"
        or (len(parsed_args) == 3 and parsed_args[2] != "--debug")
    ):
        parser.error(
            "--continue LOCATION may only be followed by the optional --debug flag"
        )

    # Keep accepting notebook-injected arguments for fresh runs.
    if unknown:
        warning(f"Running with unknown arguments: {unknown}")

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
