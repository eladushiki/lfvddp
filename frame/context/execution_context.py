from contextlib import contextmanager
from inspect import signature
from logging import basicConfig, info
import logging
import random
from configs.x_validate import cross_validate
from data_tools.dataset_config import DatasetConfig
from data_tools.detector.detector_config import DetectorConfig
from frame.cluster.cluster_config import ClusterConfig
from frame.file_system.training_history import save_training_history
from numpy import random as nprandom
from matplotlib.figure import Figure
from frame.config_handle import UserConfig
from frame.file_system.image_storage import save_figure
from frame.file_system.textual_data import load_dict_from_json, save_dict_to_json
from frame.file_structure import CONTEXT_FILE_NAME, TRAINING_OUTCOMES_DIR_NAME
from frame.context.execution_products import ExecutionProducts, stamp_product_path
from frame.git_tools import get_commit_hash, is_git_head_clean
from frame.time_tools import get_time_and_date_string, get_unix_timestamp
from plot.plotting_config import PlottingConfig
from tensorflow.keras.models import Model # type: ignore
from tensorflow import random as tfrandom

from dataclasses import dataclass, field
from os import getpid, makedirs, sep
from pathlib import Path
from sys import argv
from typing import Any, Dict, List, Optional

from train.train_config import TrainConfig


def create_config_from_paramters(
        config_params: dict,
        is_plot: bool = True,
        out_dir: Optional[str] = None,
        plot_in_place: bool = False,
):

    # Resolve config typing according to deepest hierarchy:
    config_classes = [
        ClusterConfig,
        DatasetConfig,
        DetectorConfig,
        TrainConfig,
        UserConfig,
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
            
            # Cross validate configuration
            cross_validate(self)

    # Configuration according to arguments
    if out_dir:
        config_params["config__out_dir"] = out_dir
    if plot_in_place:
        config_params["plot__target_run_parent_directory"] = config_params["config__out_dir"]

    config = DynamicConfig(**config_params)

    return config


@dataclass
class ExecutionContext:
    run_hash: str = field(init=False)
    commit_hash: str
    config: UserConfig
    command_line_args: List[str]
    time: str = get_time_and_date_string()
    random_seed: int = get_unix_timestamp() ^ (getpid() << 5)
    is_debug_mode: bool = False
    run_successful: bool = False
    products: ExecutionProducts = field(default=ExecutionProducts())
    is_reloaded: bool = False

    def __post_init__(self):
        # Run identification
        self.run_hash = hash(self._unique_descriptor)

        # Initialize once unique output directory
        if not self.is_reloaded:
            makedirs(self.unique_out_dir, exist_ok=False)

        # Random seeding
        random.seed(self.random_seed)
        nprandom.seed(self.random_seed)
        tfrandom.set_seed(self.random_seed)

    @property
    def _unique_descriptor(self) -> str:
        running_file = argv[0].split(sep)[-1]
        process_id = getpid()
        return f"run_at_{self.time}_of_{running_file}_on_commit_{self.commit_hash[:5]}_pid_{process_id}"

    @property
    def unique_out_dir(self) -> Path:
        return Path(self.config.config__out_dir) / self._unique_descriptor

    @property
    def training_outcomes_dir(self) -> Path:
        return self.unique_out_dir / TRAINING_OUTCOMES_DIR_NAME

    def document_created_product(self, product_descriptor: Any):
        self.products.add_product(product_descriptor)
        info(f"Documented product: {product_descriptor}")

    @staticmethod
    def serialize(object) -> dict:
        series = object.__dict__.copy()

        # Convert non-serializable objects
        for key, value in series.items():
            if isinstance(value, Path):
                series[key] = str(value)
            elif isinstance(value, UserConfig):
                series[key] = ExecutionContext.serialize(value)

        return series

    def _run_stamp_product_path(self, file_path: Path) -> Path:
        return stamp_product_path(file_path, self.run_hash)

    # todo: export to decorator and add os.makedirs(out_dir, exist_ok=False)
    def save_and_document_dict(self, dict: dict, file_path: Path) -> Path:
        file_path = self._run_stamp_product_path(file_path)
        save_dict_to_json(dict, file_path)
        self.document_created_product(file_path)
        return file_path

    def save_and_document_figure(self, figure: Figure, file_path: Path) -> Path:
        file_path = self._run_stamp_product_path(file_path)
        save_figure(figure, file_path)
        self.document_created_product(file_path)
        return file_path

    def save_and_document_text(self, text: str, file_path: Path) -> Path:
        file_path = self._run_stamp_product_path(file_path)
        with open(file_path, 'w') as file:
            file.write(text)
        self.document_created_product(file_path)
        return file_path

    def save_and_document_model_weights(self, model: Model, file_path: Path) -> Path:
        file_path = self._run_stamp_product_path(file_path)
        model.save_weights(file_path)
        self.document_created_product(file_path)
        return file_path

    def save_and_document_model_history(
            self,
            model_history: Dict[str, Any],
            file_path: Path,
        ):
        file_path = self._run_stamp_product_path(file_path)
        save_training_history(
            model_history,
            file_path,
            self.config.train__epochs,
            epochs_checkpoint=self.config.train__number_of_epochs_for_checkpoint,
        )
        self.document_created_product(file_path)

    def close(self):
        self.run_successful = True
        self.save_self_to_out_file()

    def save_self_to_out_file(self) -> None:
        save_dict_to_json(ExecutionContext.serialize(self), self.unique_out_dir / CONTEXT_FILE_NAME)

    @classmethod
    def naive_load_from_file(cls, file_path: Path) -> 'ExecutionContext':
        """
        Load the context from a file. Does not create classes
        from data, and currently only allows probing saved
        parameters.
        """
        data = load_dict_from_json(file_path)
        data["config"] = create_config_from_paramters(data["config"])
        data["is_reloaded"] = True
        context = cls(**data)

        return context


@contextmanager
def version_controlled_execution_context(config: UserConfig, command_line_args: List[str], is_debug_mode: bool = False):
    """
    Create a context which should contain any run dependent information.
    The data is later stored in the output_path for documentation.
    """
    # Force run on strict commit
    if not is_debug_mode and not is_git_head_clean():
        raise RuntimeError("Commit changes before running the script.")

    # Initialize
    context = ExecutionContext(get_commit_hash(), config, command_line_args, is_debug_mode=is_debug_mode)

    # Save in case run terminates prematurely
    context.save_self_to_out_file()
    basicConfig(level=getattr(logging, config.config__log_level))

    # Do everything, add important stuff as parameters to context object
    yield context

    # Overwrite saved context at end of run
    context.close()
