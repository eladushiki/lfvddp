from inspect import signature
from sys import argv
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
from typing import Callable

from data_tools.event_generation.dataset_configs import DatasetConfigs
from frame.config_handle import UserConfig
from frame.context.execution_context import version_controlled_execution_context
from frame.file_system.textual_data import load_dict_from_json
from plot.plotting_config import PlottingConfig
from frame.cluster.cluster_config import ClusterConfig
from train.train_config import TrainConfig


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

    args = parser.parse_args()

    # Parse configuration files
    config_paths = [
        args.user_config_path,
        args.cluster_config_path,
        args.dataset_config_path,
        args.train_config_path,
    ]
    if args.plot_config_path:
        config_paths.append(args.plot_config_path)

    config_params = {}
    for config_path in config_paths:
        config_params.update(load_dict_from_json(config_path))

    # Resolve config typing according to deepest hierarchy:
    config_classes = [
        UserConfig,
        ClusterConfig,
        TrainConfig,
    ]
    config_classes.append(
        DatasetConfigs[config_params["dataset__background_data_generation_function"]]
    )
    if args.plot_config_path:
        config_classes.append(PlottingConfig)

    class DynamicConfig(*config_classes):
        def __init__(self, **kwargs):
            for config_class in config_classes:
                filtered_args = {
                    k: v for k, v in kwargs.items()
                    if k in signature(config_class).parameters
                }
                config_class.__init__(self, **filtered_args)

    config = DynamicConfig(**config_params)

    # Configuration according to arguments
    is_debug_mode = args.debug
    if args.out_dir:
        config.config__out_dir = args.out_dir

    @wraps(function)
    def context_controlled_function(*args, **kwargs):
        """
        Run any decorated function in this run with the documentation of the
        configuration file parsed above.
        """
        with version_controlled_execution_context(config, argv, is_debug_mode) as context:
            function(*args, **kwargs, context=context)

    return context_controlled_function
