from logging import error, info

from aiohttp import ClientError
from data_tools.data_utils import DataSet
import numpy as np
from numpy.exceptions import AxisError
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import awkward as ak
from numpy.typing import NDArray
import uproot


def load_root_events(
    XRootD_url: str,
    tree_key: str = "Events",
    branch_names: List[str] = [],
    cut: Optional[str] = None,
    aliases: Optional[Dict[str, str]] = None,
    start: int = 0,
    stop: Optional[int] = None,
    step_size: int = 1000,
    network_retries: int = 3,
) -> DataSet:
    
    # Load events from a ROOT file
    with uproot.open(XRootD_url) as file:

        # Find the tree root from withing the file
        try:
            tree = file[tree_key]

        except KeyError as ke:
            error(f"KeyError: The key '{tree_key}' does not exist in the ROOT file at {XRootD_url}.")
            raise ke
        
        # If no branch names are provided, read all branches
        if not branch_names:
            branch_names = tree.keys()

        data_sets = []
        iter = 0

        # Generate global maxima for numbers of parameters
        length_maxima = []
        for name in branch_names:
            
            for i in range(network_retries):
                try:
                    branch = tree[name].array()
                    break
                except (ClientError, TimeoutError):
                    if i == network_retries - 1:
                        raise ConnectionError("Failed to retrieve branch data after multiple attempts.")

            if branch.ndim == 1:
                length_maxima.append(1)
            else:
                length_maxima.append(max(ak.num(branch)))
            
        # Load the desired range by batches
        for batch in tree.iterate(
            step_size=step_size,
            filter_branch=lambda TBranch: TBranch.name in branch_names,
            cut=cut,
            aliases=aliases,
            entry_start=start,
            entry_stop=stop,
            library="ak",
        ):
            data_sets.append(__branches_to_events(
                batch,
                branch_names,
                length_maxima,
            ))
            info(f"Read {batch.count} items from {tree_key}")

        return sum(data_sets, DataSet())


def __branches_to_events(
        arrays: ak.Array,
        branch_names: List[str],
        length_maxima: List[int]
) -> DataSet:
    
    # Convert to padded numpy arrays and name columns accordingly
    numpy_arrays = []
    observable_names = []
    for i, field_name in enumerate(arrays.fields):
        num_columns = length_maxima[i]

        # Convert and pad
        numpy_array = __pad_awkward_array_to_numpy(
            arrays[field_name],
            num_columns,
        )
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

    return DataSet(events, observable_names=observable_names)


def __pad_awkward_array_to_numpy(
    ak_array: ak.Array,
    length_max: int,
    default_value: Any = -1,
) -> NDArray[np.float64]:
    """
    convert an awkward array to a padded numpy array.
    Returns the padded array and the number of columns.
    """
    fixed_array = ak.values_astype(ak_array, np.float64)

    # Find the maximum length across all events
    if ak_array.ndim == 1:
        np_padded_array = ak.unflatten(fixed_array, 1).to_numpy()

    else:  # ndim is 2

        # Pad with zeros and convert to numpy
        padded_array = ak.pad_none(fixed_array, target=length_max, axis=1)
        np_converted_array = ak.to_numpy(padded_array)
        
        # Replace None values with 0
        np_padded_array = np.where(np_converted_array == None, default_value, np_converted_array)

    return np_padded_array.astype(np.float64)  # Ensure numeric type


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
