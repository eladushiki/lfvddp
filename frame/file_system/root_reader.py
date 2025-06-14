from logging import error, info
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, List, Tuple
import awkward as ak
from numpy.typing import NDArray
import uproot


def load_root_events(
    XRootD_url: str,
    tree_key: str = "Events",
    branch_names: List[str] = [],
    start: int = 0,
    stop: int = 1000,
) -> Tuple[List[str], NDArray[np.float64]]:
    
    # Load events from a ROOT file
    with uproot.open(XRootD_url) as file:
        try:
            tree = file[tree_key]

        except KeyError as ke:
            error(f"KeyError: The key '{tree_key}' does not exist in the ROOT file at {XRootD_url}.")
            raise ke
        
        if not branch_names:
            # If no branch names are provided, read all branches
            branch_names = tree.keys()

        try:
            arrays = tree.arrays( # type: ignore
                branch_names,
                entry_start=start,
                entry_stop=stop,
                library="ak",
            )
            info(f"Read items from {tree_key} numbers {start} to {stop} of {tree.num_entries}")

        except KeyError as ke:
            error(f"KeyError: One or more branch names {branch_names} do not exist in the ROOT file at {XRootD_url}.")
            raise ke
        
        return __branches_to_events(arrays, branch_names)


def __branches_to_events(
        arrays: ak.Array,
        branch_names: List[str],
) -> Tuple[List[str], NDArray[np.float64]]:
    
    # Convert to padded numpy arrays and name columns accordingly
    numpy_arrays = []
    observable_names = []
    for i, field_name in enumerate(arrays.fields):

        # Convert and pad
        numpy_array, num_columns = __pad_awkward_array_to_numpy(arrays[field_name])
        numpy_arrays.append(numpy_array)

        # Duplicate observable names according to the number of columns
        branch_name = branch_names[i]
        if num_columns > 1:
            observable_names.extend([f"{branch_name}_{col}" for col in range(num_columns)])
        else:
            observable_names.append(branch_name)
    
    # Convert to a single 2D numpy array
    events = np.column_stack(numpy_arrays)

    info(f"Loaded DataSet with observables: {observable_names}")

    return observable_names, events


def __pad_awkward_array_to_numpy(
    ak_array: ak.Array,
    default_value: Any = None,
) -> Tuple[NDArray[np.float64], int]:
    """
    convert an awkward array to a padded numpy array.
    Returns the padded array and the number of columns.
    """
    # Find the maximum length across all events
    if ak_array.ndim == 1:
        np_padded_array = ak.unflatten(ak_array, 1).to_numpy()
        max_length = 1

    else:  # ndim is 2
        max_length = ak.max(ak.num(ak_array))

        # Pad with zeros and convert to numpy
        np_none_padded_array = ak.to_numpy(ak.pad_none(ak_array, max_length, clip=True))
        
        # Replace None values with 0
        np_padded_array = np.where(np_none_padded_array == None, default_value, np_none_padded_array)

    return np_padded_array.astype(np.float64), int(max_length)  # Ensure numeric type


def save_root_events(
    root_file_path: Path,
    events: NDArray[np.float64],
    observable_names: List[str],
    tree_key: str = "Events",
):
    """
    Save a 2D array of events to a ROOT file in the structure of branches.
    """
    
    branches = {
        name: ak.Array(events[:, i]) for i, name in enumerate(observable_names)
    }
    
    with uproot.recreate(root_file_path) as root_file:
        root_file[tree_key] = branches

        info(f"Saved branches {list(branches.keys())} to {root_file_path} root file.")
