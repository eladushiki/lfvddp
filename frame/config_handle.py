from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from typing_extensions import Self
from frame.file_storage import load_dict_from_json


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


def convert_string_filenames_to_paths(config_params: dict) -> None:
    """
    Convert all string filenames to Path objects.
    """
    for key, value in config_params.items():
        if isinstance(value, str) and value != "":
            if (string_as_path := Path(value)).exists():
                config_params[key] = string_as_path
