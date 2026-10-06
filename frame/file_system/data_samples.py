"""Storage for the sampled data and display bins of a training run."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_tools.data_utils import DataSet
from frame.file_structure import DATA_SAMPLES_FILE_NAME

BACKGROUND_KEY = "background"
SIGNAL_KEY = "signal"
OBSERVABLE_NAMES_KEY = "observable_names"
BIN_EDGES_KEY_PREFIX = "bin_edges_"


@dataclass(frozen=True)
class DataSamples:
    background: DataSet
    signal: DataSet
    bin_edges_by_observable: dict[str, np.ndarray]


def save_data_samples(
    run_directory: Path,
    background: DataSet,
    signal: DataSet,
    observable_names: list[str],
    bin_edges_by_observable: dict[str, np.ndarray],
) -> None:
    arrays = {
        BACKGROUND_KEY: background.filter_observable_names(observable_names).events,
        SIGNAL_KEY: (
            signal.filter_observable_names(observable_names).events
            if not signal.empty
            else np.empty((0, len(observable_names)))
        ),
        OBSERVABLE_NAMES_KEY: np.asarray(observable_names),
    }
    arrays.update(
        {
            f"{BIN_EDGES_KEY_PREFIX}{index}": bin_edges_by_observable[name]
            for index, name in enumerate(observable_names)
        }
    )
    np.savez_compressed(
        run_directory / DATA_SAMPLES_FILE_NAME,
        **arrays,
    )


def load_data_samples(run_directory: Path) -> DataSamples:
    path = run_directory / DATA_SAMPLES_FILE_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Loaded performance significance needs training samples {path}. "
            "This training run predates saved data samples."
        )
    with np.load(path, allow_pickle=False) as components:
        observable_names = list(components[OBSERVABLE_NAMES_KEY])
        return DataSamples(
            background=DataSet(components[BACKGROUND_KEY], observable_names),
            signal=DataSet(components[SIGNAL_KEY], observable_names),
            bin_edges_by_observable={
                name: components[f"{BIN_EDGES_KEY_PREFIX}{index}"]
                for index, name in enumerate(observable_names)
            },
        )
