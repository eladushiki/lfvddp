from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing import List
from typing_extensions import Self
from numpy import random

from frame.git_tools import get_commit_hash, is_git_head_clean
from frame.time_tools import get_time_and_date_string, get_unix_timestamp


@dataclass
class Config:
    """
    An extendable class for handling configuration informatino.
    The basic always-needed configuration parameters are those that are user dependent.
    Any other class that inherits from this should add its own parameters, such that 
    their existence is checked upon loading the configuration and later stored when running.
    """
    user: str
    out_dir: Path
    scripts_dir: Path
    time: str = get_time_and_date_string()
    random_seed: int = get_unix_timestamp()

    @classmethod
    def load_from_files(cls, config_paths: List[Path]) -> Self:
        config_params = {}

        for config_path in config_paths:
            with open(config_path, 'r') as file:
                config = json.load(file)
                config_params.update(config)

        return cls(**config_params)

    def save_to_file(self, config_path: Path) -> None:
        with open(config_path, 'w') as file:
            json.dump(self.__dict__, file, indent=4)


@dataclass
class ExecutionContext:
    commit_hash: str
    config: Config


@contextmanager
def version_controlled_execution_context(config: Config):
    """
    Create a context which should contain any run dependent information.
    The data is later stored in the output_path for documentatino.
    """
    if not is_git_head_clean():
        raise RuntimeError("Commit changes before running the script.")
    
    random.seed(config.random_seed)

    context = ExecutionContext(get_commit_hash(), config)
    
    yield context
    
    with open(Config.out_dir / "context.json", 'w') as file:
        json.dump(context, file, indent=4)
