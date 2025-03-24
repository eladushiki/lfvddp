from dataclasses import dataclass
from pathlib import Path


@dataclass
class UserConfig:  # todo: convert all configs to pydantic's BaseModels
    """
    The basic always-needed configuration parameters are those that are user dependent.
    """
    config__user: str
    config__out_dir: Path
