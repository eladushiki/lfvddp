from inspect import signature
from logging import warning
from sys import argv
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
from typing import Callable, Optional

from data_tools.dataset_config import DatasetConfig
from frame.config_handle import UserConfig
from frame.context.execution_context import version_controlled_execution_context
from frame.file_system.textual_data import load_dict_from_json
from plot.plotting_config import PlottingConfig
from frame.cluster.cluster_config import ClusterConfig
from train.train_config import TrainConfig


def parse_config_from_args():# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:
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
        "--cluster-config", type=Path, required=True,
        help="Path to cluster configuration file", dest="cluster_config_path"
    )
    parser.add_argument(
        "--dataset-config", type=Path, required=True,
        help="Path to dataset configuration file", dest="dataset_config_path"
    )
    parser.add_argument(
        "--train-config", type=Path, required=True,
        help="Path to training configuration file", dest="train_config_path"
    )
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
        args.user_config_path,
        args.cluster_config_path,
        args.dataset_config_path,
        args.train_config_path,
    ]
    if args.plot_config_path:
        config_paths.append(args.plot_config_path)

    return config_paths, args.plot_config_path, args.debug, args.out_dir, args.plot_in_place


def create_config_from_paths(
        config_paths: list[Path],
        is_plot: bool = True,
        out_dir: Optional[str] = None,
        plot_in_place: bool = False,
    ):
    config_params = {}
    for config_path in config_paths:
        config_params.update(load_dict_from_json(config_path))

    # Resolve config typing according to deepest hierarchy:
    config_classes = [
        UserConfig,
        ClusterConfig,
        DatasetConfig,
        TrainConfig,
    ]

    if is_plot:
        config_classes.append(PlottingConfig)

    class DynamicConfig(*config_classes):
        def __init__(self, **kwargs):
            for config_class in config_classes:
                filtered_args = {
                    k: v for k, v in kwargs.items()
                    if k in signature(config_class).parameters
                }
                config_class.__init__(self, **filtered_args)

    
    # Configuration according to arguments
    if out_dir:
        config_params["config__out_dir"] = out_dir
    if plot_in_place:
        config_params["plot__target_run_parent_directory"] = config_params["config__out_dir"]

    config = DynamicConfig(**config_params)

    return config


def context_controlled_execution(function: Callable):# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:# -> _Wrapped[Callable[..., Any], Any, Callable[..., Any], None]:
    """
    A wrapper for any entry point in the project, to ensure context control.
    """
    config_paths, is_plot, is_debug_mode, out_dir, plot_in_place = parse_config_from_args()
    config = create_config_from_paths(config_paths, is_plot, out_dir, plot_in_place)

    @wraps(function)
    def context_controlled_function(*args, **kwargs):
        """
        Run any decorated function in this run with the documentation of the
        configuration file parsed above.
        """
        with version_controlled_execution_context(config, argv, is_debug_mode) as context:
            function(*args, **kwargs, context=context)

    return context_controlled_function
