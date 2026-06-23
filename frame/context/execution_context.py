from argparse import Namespace
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
import torch
from frame.config_handle import UserConfig
from frame.file_system.image_storage import save_figure
from frame.file_system.textual_data import load_dict_from_json, save_dict_to_json
from frame.file_structure import CONTEXT_FILE_NAME, TRAINING_OUTCOMES_DIR_NAME
from frame.context.execution_products import ExecutionProducts, stamp_product_path
from frame.git_tools import get_commit_hash, is_git_head_clean
from frame.time_tools import get_time_and_date_string, get_unix_timestamp
from plot.plotting_config import PlottingConfig

from dataclasses import dataclass, field
from os import environ, getpid, makedirs, sep
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
                if hasattr(config_class, "__post_init__"):
                    config_class.__post_init__(self)
            
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
    commit_hash: str
    config: UserConfig
    config_paths: List[Path]
    command_line_args: List[str]
    run_hash: Optional[str] = None
    run_descriptor: Optional[str] = None
    time: str = get_time_and_date_string()
    random_seed: int = get_unix_timestamp() ^ (getpid() << 5)
    is_debug_mode: bool = False
    is_no_build: bool = False
    is_only_train: bool = False
    is_continue: bool = False
    continue_from: Optional[Path] = None
    array_index: Optional[int] = None
    qsub_submissions: List[Dict[str, Any]] = field(default_factory=list)
    run_successful: bool = False
    products: ExecutionProducts = field(default_factory=ExecutionProducts)
    is_reloaded: bool = False

    def __post_init__(self):
        # Run identification
        if self.run_descriptor is None:
            self.run_descriptor = self._make_unique_descriptor()
        if self.run_hash is None:
            self.run_hash = hash(self._unique_descriptor)

        if self.array_index is None:
            pbs_array_index = environ.get("PBS_ARRAY_INDEX")
            self.array_index = int(pbs_array_index) if pbs_array_index else None

        # Initialize once unique output directory
        if not self.is_reloaded:
            makedirs(self.unique_out_dir, exist_ok=False)

        # Random seeding
        random.seed(self.random_seed)
        nprandom.seed(self.random_seed)
        if self.config.train__like_NPLM:
            # NPLM's train_model uses tf, so we set its seed as well
            from tensorflow import random as tfrandom
            tfrandom.set_seed(self.random_seed)
        else:
            torch.manual_seed(self.random_seed)

    def _make_unique_descriptor(self) -> str:
        running_file = argv[0].split(sep)[-1]
        process_id = getpid()
        return f"{self.time}_{self.config.config__dirsafe_runtag}_run_of_{running_file}_pid_{process_id}"

    @property
    def _unique_descriptor(self) -> str:
        if self.run_descriptor is None:
            self.run_descriptor = self._make_unique_descriptor()
        return self.run_descriptor

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
        if isinstance(object, ClusterConfig):
            if object.cluster__qsub_total_walltime is not None:
                series["cluster__qsub_walltime"] = object.cluster__qsub_total_walltime
            series.pop("cluster__qsub_total_walltime", None)
            series.pop("cluster__qsub_walltime_chunks", None)

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

    def save_and_document_model_parameters(self, model, file_path: Path) -> Path:
        file_path = self._run_stamp_product_path(file_path)
        model.save_parameters(file_path)
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

    @property
    def qsub_submitted_chunk_count(self) -> int:
        return len(self.qsub_submissions)

    def next_qsub_walltime_chunk(self) -> Optional[str]:
        if not isinstance(self.config, ClusterConfig):
            raise TypeError(f"Expected ClusterConfig, got {self.config.__class__.__name__}")
        return self.config.next_walltime_chunk(self.qsub_submitted_chunk_count)

    def record_qsub_submission(self, walltime: str, job_id: str, submit_run_dir: Path) -> None:
        self.qsub_submissions.append({
            "chunk_index": self.qsub_submitted_chunk_count + 1,
            "walltime": walltime,
            "job_id": job_id,
            "submitted_at": get_time_and_date_string(),
            "submit_run_dir": str(submit_run_dir),
        })

    @classmethod
    def load_from_run_dir(cls, run_dir: Path) -> 'ExecutionContext':
        run_dir = Path(run_dir)
        context_path = run_dir if run_dir.name == CONTEXT_FILE_NAME else run_dir / CONTEXT_FILE_NAME
        return cls.naive_load_from_file(context_path)

    @classmethod
    def find_stamped_submit_context(
        cls,
        out_dir: Path,
        dirsafe_runtag: str,
    ) -> Optional['ExecutionContext']:
        out_dir = Path(out_dir)
        if (out_dir / CONTEXT_FILE_NAME).exists():
            return cls.load_from_run_dir(out_dir)
        if not out_dir.exists():
            return None

        candidates = list(out_dir.glob(
            f"*_{dirsafe_runtag}_run_of_submit_train.py_pid_*/{CONTEXT_FILE_NAME}"
        ))
        if not candidates:
            return None

        loaded_candidates = [
            (cls.naive_load_from_file(candidate), candidate)
            for candidate in candidates
        ]
        loaded_candidates = [
            (context, candidate)
            for context, candidate in loaded_candidates
            if isinstance(context.config, ClusterConfig) and context.config.cluster__qsub_needs_continuation
        ]
        if not loaded_candidates:
            return None

        context, _ = max(
            loaded_candidates,
            key=lambda item: (len(item[0].qsub_submissions), item[1].stat().st_mtime),
        )
        return context

    @classmethod
    def naive_load_from_file(cls, file_path: Path) -> 'ExecutionContext':
        """
        Load the context from a file. Does not create classes
        from data, and currently only allows probing saved
        parameters.
        """
        data = load_dict_from_json(file_path)
        data["config"] = create_config_from_paramters(data["config"])
        data["config_paths"] = [Path(path) for path in data.get("config_paths", [])]
        if data.get("continue_from") is not None:
            data["continue_from"] = Path(data["continue_from"])
        data["is_reloaded"] = True
        context = cls(**data)

        return context


@contextmanager
def version_controlled_execution_context(
    config: UserConfig,
    config_paths: List[Path],
    command_line_args: List[str],
    args: Namespace,
):
    """
    Create a context which should contain any run dependent information.
    The data is later stored in the output_path for documentation.
    """
    # Force run on strict commit
    if not args.debug and not is_git_head_clean():
        raise RuntimeError("Commit changes before running the script.")

    # Initialize
    context = ExecutionContext(
        get_commit_hash(),
        config,
        config_paths,
        command_line_args,
        is_debug_mode=args.debug,
        is_no_build=args.no_build,
        is_only_train=args.only_train,
        is_continue=args.continue_training,
        continue_from=args.continue_from,
    )

    # Save in case run terminates prematurely
    context.save_self_to_out_file()
    basicConfig(level=getattr(logging, config.config__log_level))

    # Do everything, add important stuff as parameters to context object
    yield context

    # Overwrite saved context at end of run
    context.close()
