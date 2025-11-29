from logging import warning
from sys import argv
from argparse import ArgumentParser, Namespace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

from frame.context.execution_context import version_controlled_execution_context
from frame.context.execution_context import create_config_from_paramters
from frame.file_system.textual_data import load_config_file


def parse_config_from_args() -> tuple[list[Any], Namespace]:
    """
    A wrapper for any entry point in the project, to ensure context control.
    """
    parser  = ArgumentParser()
        
    # Mandatory arguments
    parser.add_argument(
        "--cluster-config", type=Path, required=True,
        help="Path to cluster configuration file", dest="cluster_config_path"
    )
    parser.add_argument(
        "--dataset-config", type=Path, required=True,
        help="Path to dataset configuration file", dest="dataset_config_path"
    )
    parser.add_argument(
        "--detector-config", type=Path, required=True,
        help="Path to detector configuration file", dest="detector_config_path"
    )
    parser.add_argument(
        "--train-config", type=Path, required=True,
        help="Path to training configuration file", dest="train_config_path"
    )
    parser.add_argument(
        "--user-config", type=Path, required=True,
        help="User details configuration file path", dest="user_config_path"
    )

    # Optional arguments
    parser.add_argument(
        "--plot-config", type=Path, required=False,
        help="Path to plot configuration file", dest="plot_config_path"
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
        "--out-dir", type=str,
        help="Output directory for results. Overrides one in config file. Useful for aggregating batch jobs", dest="out_dir"
    )
    parser.add_argument(
        "--plot-in-place", action="store_true",
        help="Should create plots in the output directory? Else, in a dedicated one", dest="plot_in_place"
    )

    args, unknown = parser.parse_known_args()  # Using this instead of parse_args() to enable calling from jupyter
    if unknown:
        warning(f"Running with nknown arguments: {unknown}")

    # Parse configuration files
    config_paths = [
        args.cluster_config_path,
        args.dataset_config_path,
        args.detector_config_path,
        args.train_config_path,
        args.user_config_path,
    ]
    if args.plot_config_path:
        config_paths.append(args.plot_config_path)

    return config_paths, args


def create_config_from_paths(
        config_paths: list[Path],
        is_plot: bool = True,
        out_dir: Optional[str] = None,
        plot_in_place: bool = False,
    ):
    config_params = {}
    for config_path in config_paths:
        config_params.update(load_config_file(config_path))

    return create_config_from_paramters(
        config_params,
        is_plot=is_plot,
        out_dir=out_dir,
        plot_in_place=plot_in_place,
    )

def context_controlled_execution(function: Callable):# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:
    """
    A wrapper for any entry point in the project, to ensure context control.
    """
    config_paths, args = parse_config_from_args()
    config = create_config_from_paths(
        config_paths,
        args.plot_config_path,
        args.out_dir,
        args.plot_in_place
    )

    @wraps(function)
    def context_controlled_function(*inner_args, **inner_kwargs):
        """
        Run any decorated function in this run with the documentation of the
        configuration file parsed above.
        """
        with version_controlled_execution_context(config, argv, args) as context:
            function(*inner_args, **inner_kwargs, context=context)

    return context_controlled_function
