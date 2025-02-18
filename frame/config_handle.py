from abc import ABC
from contextlib import contextmanager
from dataclasses import dataclass
import random
from os import makedirs, getpid
from pathlib import Path
from sys import argv
from typing import Any, Dict, List
from typing_extensions import Self
from frame.file_storage import load_dict_from_json, save_dict_to_json
from numpy import random as npramdom

from frame.file_structure import CONTEXT_FILE_NAME
from frame.git_tools import get_commit_hash, is_git_head_clean
from frame.time_tools import get_time_and_date_string, get_unix_timestamp


@dataclass
class Config(ABC):
    """
    An extendable class for handling configuration informatino.
    The basic always-needed configuration parameters are those that are user dependent.
    Any other class that inherits from this should add its own parameters, such that 
    their existence is checked upon loading the configuration and later stored when running.

    How to inherit this class to create a new config?
    1. Create a new class that inherits from Config *and ABC*. It may inherit from a subclass to enforce the inclusion of that one as well.
    2. Add an appropriate argument in "handle_args.py" to include the new class interface.
    2. A class instance is dynamically created each run according to the input config files.
    """
    user: str
    out_dir: Path

    @classmethod
    def load_from_files(cls, config_paths: List[Path]) -> Self:
        config_params = {}

        for config_path in config_paths:
            config_params.update(load_dict_from_json(config_path))

        convert_string_filenames_to_paths(config_params)

        resolved_class = cls.dynamic_class_resolve(config_params)

        return resolved_class(**config_params)

    @classmethod
    def dynamic_class_resolve(cls, config_params: Dict[str, Any]):
        # Override hook to give a subclass that is dynamically resolved
        return cls

    def save_to_file(self, config_path: Path) -> None:
        save_dict_to_json(self.__dict__, config_path)


def convert_string_filenames_to_paths(config_params: dict) -> None:
    """
    Convert all string filenames to Path objects.
    """
    for key, value in config_params.items():
        if isinstance(value, str) and value != "":
            if (string_as_path := Path(value)).exists():
                config_params[key] = string_as_path


@dataclass
class ExecutionContext:
    commit_hash: str
    config: Config
    command_line_args: List[str]
    time: str = get_time_and_date_string()
    random_seed: int = get_unix_timestamp()
    is_debug_mode: bool = False
    run_successful: bool = False

    def __post_init__(self):
        # Initialize once unique output directory
        makedirs(self.unique_out_dir, exist_ok=False)

    @property
    def unique_descriptor(self) -> str:
        running_file = argv[0].split('/')[-1]
        process_id = getpid()
        return f"run_at_{self.time}_of_{running_file}_on_commit_{self.commit_hash[:5]}_pid_{process_id}"

    @property
    def unique_out_dir(self) -> Path:
        return self.config.out_dir / self.unique_descriptor

    @staticmethod
    def serialize(object) -> dict:
        series = object.__dict__.copy()

        # Convert non-serializable objects
        for key, value in series.items():
            if isinstance(value, Path):
                series[key] = str(value)
            elif isinstance(value, Config):
                series[key] = ExecutionContext.serialize(value)

        return series
    
    def save_to_out_file(self) -> None:
        save_dict_to_json(ExecutionContext.serialize(self), self.unique_out_dir / CONTEXT_FILE_NAME)

@contextmanager
def version_controlled_execution_context(config: Config, command_line_args: List[str], is_debug_mode: bool = False):
    """
    Create a context which should contain any run dependent information.
    The data is later stored in the output_path for documentatino.
    """
    # Force run on strict commit
    if not is_debug_mode and not is_git_head_clean():
        raise RuntimeError("Commit changes before running the script.")
    
    # Initialize
    context = ExecutionContext(get_commit_hash(), config, command_line_args, is_debug_mode=is_debug_mode)
    random.seed(context.random_seed)
    npramdom.seed(context.random_seed)

    # Save in case run terminates prematurely
    context.save_to_out_file()
    
    # Do everyting, add imoprttant stufff as parameters to context object
    yield context

    # Overwrite saved context at end of run
    context.run_successful = True    
    context.save_to_out_file()
