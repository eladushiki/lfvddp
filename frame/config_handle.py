from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
from typing_extensions import Self

from frame.git_tools import get_commit_hash, is_git_head_clean

@dataclass
class Config:

    @classmethod
    def load_from_file(cls, config_path: Path) -> Self:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return cls(**config)

    def save_to_file(self, config_path: Path) -> None:
        with open(config_path, 'w') as file:
            json.dump(self.__dict__, file, indent=4)


@dataclass
class ExecutionContext:
    commit_hash: str
    config: Config


@contextmanager
def version_controlled_execution(output_path: Path, config: Config):
    """
    Create a context which should contain any run dependent information.
    The data is later stored in the output_path for documentatino.
    """
    if not is_git_head_clean():
        raise RuntimeError("Commit changes before running the script.")
    context = ExecutionContext(get_commit_hash(), config)
    
    yield context
    
    with open(output_path / "context.json", 'w') as file:
        json.dump(context, file, indent=4)
