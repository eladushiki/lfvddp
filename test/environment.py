from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union


class ConfigType(Enum):
    CLUSTER = "cluster"
    DATASET = "dataset"
    DETECTOR = "detector"
    PLOT = "plot"
    TRAIN = "train"
    USER = "user"


@dataclass(frozen=True)
class TrainConfigFixture:
    """Materialize a per-test train configuration instead of tracking run packs."""

    values: Mapping[str, Any]

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "train_config.json"
        path.write_text(json.dumps(deepcopy(dict(self.values))))
        return path


DEFAULT_CONFIG_PATHS = {t: Path(s) for t, s in zip(
    ConfigType,
    [
        "configs/basic-loaded/cluster_config.json",
        "configs/basic-loaded/loaded_dataset_config.json",
        "configs/basic-loaded/detector_config.json",
        "configs/basic-loaded/plot_config.json",
        "configs/basic-loaded/train_config.json",
        "configs/basic-loaded/user_config.json",
    ]
)}


def wrap_with_command_line_args(
        kwconfs: Dict[Union[str, ConfigType], Path]
) -> list[str]:
    try:
        for key in kwconfs:
            ConfigType(key)
    except ValueError:
        raise ValueError("Invalid conf type")
    return ["--configs", *(str(value) for value in kwconfs.values())]
