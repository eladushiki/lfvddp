from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict


@dataclass
class UserConfig:  # todo: convert all configs to pydantic's BaseModels
    """
    The basic always-needed configuration parameters are those that are user dependent.
    """
    config__user: str
    config__out_dir: Path
    config__log_level: str
    config__bind_directories: Dict[Path, PurePosixPath]

    def __post_init__(self):
        if isinstance(self.config__out_dir, str):
            self.config__out_dir = Path(self.config__out_dir).absolute()

        converted_bind_dirs = {}
        for local, contained in self.config__bind_directories.items():
            if isinstance(local, str):
                converted_local = Path(local).absolute()
                converted_bind_dirs[converted_local] = contained
            else:
                converted_local = local.absolute()
                converted_bind_dirs[converted_local] = contained
            if isinstance(contained, str):
                converted_bind_dirs[converted_local] = PurePosixPath(contained)
        self.config__bind_directories = converted_bind_dirs
