"""Storage for the sampled components used by loaded performance plots."""

from pathlib import Path

import numpy as np

from data_tools.data_utils import DataSet
from frame.file_structure import PERFORMANCE_COMPONENTS_FILE_NAME


def save_performance_components(
    run_directory: Path,
    background: DataSet,
    signal: DataSet,
    observable_names: list[str],
) -> None:
    np.savez_compressed(
        run_directory / PERFORMANCE_COMPONENTS_FILE_NAME,
        background=background.filter_observable_names(observable_names).events,
        signal=(
            signal.filter_observable_names(observable_names).events
            if not signal.empty
            else np.empty((0, len(observable_names)))
        ),
        observable_names=np.asarray(observable_names),
    )


def load_performance_components(run_directory: Path) -> tuple[DataSet, DataSet]:
    path = run_directory / PERFORMANCE_COMPONENTS_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Loaded performance significance needs training sample {path}. "
            "This training run predates saved performance components."
        )
    with np.load(path, allow_pickle=False) as components:
        observable_names = list(components["observable_names"])
        return (
            DataSet(components["background"], observable_names),
            DataSet(components["signal"], observable_names),
        )
