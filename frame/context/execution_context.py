import logging
import random
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass, field
from inspect import signature
from logging import basicConfig, info
from os import environ, getpid, makedirs
from pathlib import Path
from sys import argv
from typing import Any, Dict, List, Optional, Tuple

import torch
from matplotlib.figure import Figure
from numpy import random as nprandom

from configs.x_validate import cross_configure, cross_validate
from data_tools.dataset_config import DatasetConfig
from data_tools.detector.detector_config import DetectorConfig
from frame.cluster.cluster_config import ClusterConfig
from frame.cluster.walltime import parse_walltime
from frame.config_handle import UserConfig
from frame.context.execution_products import ExecutionProducts, stamp_product_path
from frame.context.run_descriptor import (
    build_run_descriptor,
    context_glob_for_run,
    run_descriptor_matches,
)
from frame.file_structure import (
    CONTEXT_FILE_NAME,
    SUBMIT_TRAIN_SCRIPT_NAME,
    TRAINING_OUTCOMES_DIR_NAME,
)
from frame.file_system.image_storage import save_figure
from frame.file_system.textual_data import (
    load_config_params_from_paths,
    load_dict_from_json,
    save_dict_to_json,
)
from frame.file_system.training_history import save_training_history
from frame.git_tools import get_commit_hash, is_git_head_clean
from frame.time_tools import (
    get_time_and_date_string,
    get_unique_run_dir_name,
    get_unix_timestamp,
)
from plot.plotting_config import PlottingConfig
from train.train_config import TrainConfig


def _array_index_from_environment() -> Optional[int]:
    pbs_array_index = environ.get("PBS_ARRAY_INDEX")
    return int(pbs_array_index) if pbs_array_index else None


def create_config_from_paramters(
    config_params: dict,
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
        PlottingConfig,
    ]

    class DynamicConfig(*config_classes):
        def __init__(self, **kwargs):
            for config_class in config_classes:
                filtered_args = {
                    k: v
                    for k, v in kwargs.items()
                    if k in signature(config_class).parameters
                }
                config_class.__init__(self, **filtered_args)
                if hasattr(config_class, "__post_init__"):
                    config_class.__post_init__(self)

            # Configure and validate values that depend on the merged config.
            cross_configure(self)
            cross_validate(self)

    # Configuration according to arguments
    if out_dir:
        config_params["config__out_dir"] = out_dir
    if plot_in_place:
        config_params["plot__target_run_parent_directory"] = config_params[
            "config__out_dir"
        ]

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
    time: str = field(default_factory=get_unique_run_dir_name)
    random_seed: int = field(
        default_factory=lambda: get_unix_timestamp() ^ (getpid() << 5)
    )
    is_debug_mode: bool = False
    is_no_build: bool = False
    is_only_train: bool = False
    is_continue: bool = False
    continue_from: Optional[Path] = None
    array_index: Optional[int] = None
    qsub_submissions: List[Dict[str, Any]] = field(default_factory=list)
    qsub_walltime_chunk: Optional[str] = None
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
            self.array_index = _array_index_from_environment()

        # Initialize once unique output directory
        if not self.is_reloaded:
            makedirs(self.unique_out_dir, exist_ok=False)

        self.seed_random_generators()

    def seed_random_generators(self) -> None:
        """Seed every random backend used by the configured training path."""
        random.seed(self.random_seed)
        nprandom.seed(self.random_seed)
        if self.config.train__like_NPLM:
            # NPLM's train_model uses tf, so we set its seed as well
            from tensorflow import random as tfrandom

            tfrandom.set_seed(self.random_seed)
        else:
            torch.manual_seed(self.random_seed)

    def _make_unique_descriptor(self) -> str:
        return build_run_descriptor(
            stamp=self.time,
            dirsafe_runtag=self.config.config__dirsafe_runtag,
            entrypoint=Path(argv[0]).name,
            pid=getpid(),
        )

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
        with open(file_path, "w") as file:
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
        save_dict_to_json(
            ExecutionContext.serialize(self), self.unique_out_dir / CONTEXT_FILE_NAME
        )

    @property
    def qsub_submitted_chunk_count(self) -> int:
        return len(self.qsub_submissions)

    @property
    def qsub_submitted_walltime_seconds(self) -> int:
        return sum(
            parse_walltime(submission["walltime"])
            for submission in self.qsub_submissions
        )

    def next_qsub_walltime_chunk(self) -> Optional[str]:
        if not isinstance(self.config, ClusterConfig):
            raise TypeError(
                f"Expected ClusterConfig, got {self.config.__class__.__name__}"
            )
        return self.config.next_walltime_chunk(
            self.qsub_submitted_walltime_seconds
        )

    def add_qsub_walltime(self, extra_walltime: str) -> None:
        """Extend the total cluster walltime budget."""
        if not isinstance(self.config, ClusterConfig):
            raise TypeError(
                f"Expected ClusterConfig, got {self.config.__class__.__name__}"
            )
        self.config.add_walltime(extra_walltime)

    def prepare_next_qsub_walltime_chunk(self) -> str:
        """Select and retain the chunk attempted by this submit invocation."""
        if self.qsub_walltime_chunk is None:
            self.qsub_walltime_chunk = self.next_qsub_walltime_chunk()
        if self.qsub_walltime_chunk is None:
            raise RuntimeError(
                f"No remaining walltime chunks to submit for {self.unique_out_dir}."
            )
        self.config.use_walltime_chunk(self.qsub_walltime_chunk)
        return self.qsub_walltime_chunk

    def record_qsub_submission(
        self, walltime: str, job_id: str, submit_run_dir: Path
    ) -> None:
        self.qsub_submissions.append({
            "chunk_index": self.qsub_submitted_chunk_count + 1,
            "walltime": walltime,
            "job_id": job_id,
            "submitted_at": get_time_and_date_string(),
            "submit_run_dir": str(submit_run_dir),
        })
        self.qsub_walltime_chunk = None

    @classmethod
    def load_from_run_dir(cls, run_dir: Path) -> "ExecutionContext":
        run_dir = Path(run_dir)
        context_path = (
            run_dir
            if run_dir.name == CONTEXT_FILE_NAME
            else run_dir / CONTEXT_FILE_NAME
        )
        return cls.naive_load_from_file(context_path)

    @classmethod
    def discover_run_contexts(
        cls,
        parent_directory: Path,
        entrypoint: Optional[str] = None,
        dirsafe_runtag: Optional[str] = None,
        require_continuation: bool = False,
    ) -> List[Tuple["ExecutionContext", Path]]:
        parent_directory = Path(parent_directory)
        if not parent_directory.exists():
            return []

        if parent_directory.name == CONTEXT_FILE_NAME:
            context_paths = [parent_directory]
        elif parent_directory.is_file():
            return []
        else:
            context_paths = list(parent_directory.rglob(CONTEXT_FILE_NAME))

        contexts = []
        for context_path in context_paths:
            context = cls.naive_load_from_file(context_path)
            if not run_descriptor_matches(
                context.run_descriptor,
                entrypoint=entrypoint,
                dirsafe_runtag=dirsafe_runtag,
            ):
                continue
            if require_continuation and not (
                isinstance(context.config, ClusterConfig)
                and context.config.cluster__qsub_needs_continuation
            ):
                continue
            contexts.append((context, context_path))

        return contexts

    @classmethod
    def find_stamped_run_context(
        cls,
        out_dir: Path,
        dirsafe_runtag: str,
        entrypoint: str,
        require_continuation: bool = False,
    ) -> Optional["ExecutionContext"]:
        out_dir = Path(out_dir)
        if not out_dir.exists():
            return None

        direct_context_path = (
            out_dir
            if out_dir.name == CONTEXT_FILE_NAME
            else out_dir / CONTEXT_FILE_NAME
        )
        candidate_paths = []
        if direct_context_path.exists():
            candidate_paths.append(direct_context_path)
        if out_dir.is_dir():
            candidate_paths.extend(
                out_dir.glob(context_glob_for_run(dirsafe_runtag, entrypoint))
            )

        candidates = []
        seen_paths = set()
        for context_path in candidate_paths:
            if context_path in seen_paths:
                continue
            seen_paths.add(context_path)
            context = cls.naive_load_from_file(context_path)
            if not run_descriptor_matches(
                context.run_descriptor,
                entrypoint=entrypoint,
                dirsafe_runtag=dirsafe_runtag,
            ):
                continue
            if require_continuation and not (
                isinstance(context.config, ClusterConfig)
                and context.config.cluster__qsub_needs_continuation
            ):
                continue
            candidates.append((context, context_path))

        if not candidates:
            return None

        context, _ = max(
            candidates,
            key=lambda item: (len(item[0].qsub_submissions), item[1].stat().st_mtime),
        )
        return context

    @classmethod
    def naive_load_from_file(cls, file_path: Path) -> "ExecutionContext":
        """
        Load the context from a file. Does not create classes
        from data, and currently only allows probing saved
        parameters.
        """
        data = load_dict_from_json(file_path)
        random_seed = data.pop("random_seed")
        data["config"] = create_config_from_paramters(data["config"])
        data["config_paths"] = [Path(path) for path in data.get("config_paths", [])]
        if data.get("continue_from") is not None:
            data["continue_from"] = Path(data["continue_from"])
        data["products"] = ExecutionProducts.from_serialized(data.get("products", {}))
        data["is_reloaded"] = True
        context = cls(random_seed=random_seed, **data)

        return context

    @classmethod
    def load_child_run_context(
        cls,
        parent_directory: Path,
        entrypoint: str,
        array_index: Optional[int],
    ) -> "ExecutionContext":
        """Load the matching worker context below a submitted run."""
        candidates = []
        for candidate, context_path in cls.discover_run_contexts(
            parent_directory,
            entrypoint=entrypoint,
        ):
            if candidate.array_index == array_index:
                candidates.append((candidate, context_path))

        if not candidates:
            raise RuntimeError(
                f"Could not find a prior {Path(entrypoint).name} context to continue "
                f"for array index {array_index} below {parent_directory}."
            )

        context, _ = max(
            candidates,
            key=lambda item: (
                len(item[0].qsub_submissions),
                item[1].stat().st_mtime,
            ),
        )
        return context


@contextmanager
def version_controlled_execution_context(
    config: Optional[UserConfig],
    config_paths: Optional[List[Path]],
    command_line_args: List[str],
    args: Namespace,
):
    """
    Create a context which should contain any run dependent information.
    The data is later stored in the output_path for documentation.
    """
    array_index = _array_index_from_environment()
    entrypoint = Path(command_line_args[0] if command_line_args else argv[0]).name
    if args.continue_from is not None:
        continue_from = Path(args.continue_from)
        context_path = (
            continue_from
            if continue_from.name == CONTEXT_FILE_NAME
            else continue_from / CONTEXT_FILE_NAME
        )
        context = None
        if context_path.exists():
            directly_loaded_context = ExecutionContext.naive_load_from_file(context_path)
            if (
                directly_loaded_context.array_index == array_index
                and run_descriptor_matches(
                    directly_loaded_context.run_descriptor,
                    entrypoint=entrypoint,
                )
            ):
                context = directly_loaded_context
        if context is None:
            context = ExecutionContext.load_child_run_context(
                parent_directory=continue_from,
                entrypoint=entrypoint,
                array_index=array_index,
            )
        if args.debug:
            context.is_debug_mode = True
        if not context.is_debug_mode and not is_git_head_clean():
            raise RuntimeError("Commit changes before running the script.")

        extra_time = getattr(args, "extra_time", None)
        if extra_time is not None:
            context.add_qsub_walltime(extra_time)
            cross_validate(context.config)

        context.is_continue = True
        context.continue_from = args.continue_from
        context.run_successful = False
        context.seed_random_generators()
    else:
        if config is None or config_paths is None:
            raise ValueError("Fresh runs require configuration files.")
        if not args.debug and not is_git_head_clean():
            raise RuntimeError("Commit changes before running the script.")

        random_seed = load_config_params_from_paths(config_paths).get("random_seed")
        context_kwargs = {}
        if random_seed is not None:
            context_kwargs["random_seed"] = random_seed

        context = ExecutionContext(
            get_commit_hash(),
            config,
            config_paths,
            command_line_args,
            is_debug_mode=args.debug,
            is_no_build=args.no_build,
            is_only_train=args.only_train,
            array_index=array_index,
            **context_kwargs,
        )

    if entrypoint == SUBMIT_TRAIN_SCRIPT_NAME:
        context.prepare_next_qsub_walltime_chunk()

    # Save in case run terminates prematurely
    context.save_self_to_out_file()
    basicConfig(level=getattr(logging, context.config.config__log_level))

    # Do everything, add important stuff as parameters to context object
    yield context

    # Overwrite saved context at end of run
    context.close()
