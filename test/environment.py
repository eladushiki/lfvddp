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


CONFIG_ARGUMENTS = [
    (ConfigType.CLUSTER, "--cluster-config"),
    (ConfigType.DATASET, "--dataset-config"),
    (ConfigType.DETECTOR, "--detector-config"),
    (ConfigType.TRAIN, "--train-config"),
    (ConfigType.USER, "--user-config"),
]


DEFAULT_CONFIG_PATHS = {t.value: Path(s) for t, s in zip(
    ConfigType,
    [
        "configs/cluster/basic_cluster_config.json",
        "configs/dataset/basic_loaded_dataset_config.json",
        "configs/detector/basic_detector_config.json",
        "configs/plot/basic_plot_config.json",
        "configs/train/basic_train_config.json",
        "configs/user/basic_user_config.json",
    ]
)}


def wrap_with_command_line_args(
        kwconfs: Dict[Union[str, ConfigType], Path]
) -> list[str]:
    try:
        return [f"--{ConfigType(key).value}-config {value}" for key, value in kwconfs.items()]
    except ValueError:
        raise ValueError("Invalid conf type")
