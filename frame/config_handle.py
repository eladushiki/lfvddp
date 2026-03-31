from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict
from werkzeug.utils import secure_filename


@dataclass
class UserConfig:  # todo: convert all configs to pydantic's BaseModels
    """
    The basic always-needed configuration parameters are those that are user dependent.
    """
    config__user: str
    config__runtag: str
    config__out_dir: Path
    config__log_level: str
    config__bind_directories: Dict[Path, PurePosixPath]

    @property
    def config__dirsafe_runtag(self) -> str:
        return secure_filename(self.config__runtag)
