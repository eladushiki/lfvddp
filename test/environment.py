from enum import Enum
from pathlib import Path
from typing import Dict, Union


class ConfigType(Enum):
    CLUSTER = "cluster"
    DATASET = "dataset"
    DETECTOR = "detector"
    PLOT = "plot"
    TRAIN = "train"
    USER = "user"


DEFAULT_CONFIG_PATHS = {t: Path(s) for t, s in zip(
    ConfigType,
    [
        "configs/basic-loaded/basic_cluster_config.json",
        "configs/basic-loaded/basic_loaded_dataset_config.json",
        "configs/basic-loaded/basic_detector_config.json",
        "configs/basic-loaded/basic_plot_config.json",
        "configs/basic-loaded/basic_train_config.json",
        "configs/basic-loaded/basic_user_config.json",
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
