from sys import argv
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
from typing import Callable

from frame.config_handle import Config, version_controlled_execution_context
from train.train_config import ClusterConfig, TrainConfig


def context_controlled_execution(function: Callable):# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:
    """
    A wrapper for any entry point in the project, to ensure context control.
    """
    parser  = ArgumentParser()
        
    # Mandatory arguments
    parser.add_argument(
        "--user-config", type=Path, required=True,
        help="User details configuration file path", dest="user_config_path"
    )

    # Optional arguments
    ## Additional configurations
    parser.add_argument(
        "--cluster-config", type=Path, required=False,
        help="Path to cluster configuration file", dest="cluster_config_path"
    )
    parser.add_argument(
        "--train-config", type=Path, required=False,
        help="Path to training configuration file", dest="train_config_path"
    )
    ## Running options
    parser.add_argument(
        "--debug", action="store_true",
        help="Run in debug mode. NOTE: Does not verify running on strict commits"
    )
    parser.add_argument(
        "--out-dir", type=Path,
        help="Output directory for results. Overrides one in config file. Useful for aggregating batch jobs", dest="out_dir"
    )

    args = parser.parse_args()

    # Parse configuration files
    config_paths = [args.user_config_path]
    config_class = Config
    if args.cluster_config_path:
        config_paths.append(args.cluster_config_path)
        config_class = ClusterConfig
    if args.train_config_path:
        config_paths.append(args.train_config_path)
        config_class = TrainConfig
    config = config_class.load_from_files(config_paths)

    # Configuration according to arguments
    is_debug_mode = args.debug
    if args.out_dir:
        config.out_dir = Path(args.out_dir)

    @wraps(function)
    def context_controlled_function(*args, **kwargs):
        """
        Run any decorated function in this run with the documentation of the
        confguration file parsed above.
        """
        with version_controlled_execution_context(config, argv, is_debug_mode) as context:
            function(*args, **kwargs, context=context)

    return context_controlled_function
