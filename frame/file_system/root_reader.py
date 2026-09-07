from logging import error, info

import pandas as pd
from data_tools.data_utils import DataSet
import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Dict, List, Optional, Union
import awkward as ak
import awkward_pandas  # noqa: F401  # registers pandas' Awkward extension dtype
from numpy.typing import NDArray
import uproot


def load_root_events(
    XRootD_url: Union[str, List[str]],
    tree_key: str = "Events",
    branch_names: List[str] = [],
    observable_renames: Dict[str, str] = {},
    cut: Optional[str] = None,
    aliases: Optional[Dict[str, str]] = None,
    start: int = 0,
    stop: Optional[int] = None,
    step_size: int = 1000,
) -> DataSet:
    
    if isinstance(XRootD_url, list):
        XRootD_url = uproot.concatenate([f"{url}:{tree_key}" for url in XRootD_url])
    
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

        # Load the desired range by batches.  Uproot's pandas backend cannot
        # apply a jagged particle-level mask to event-level data, so evaluate
        # cuts with Awkward and reduce them to one boolean per event first.
        if cut is None:
            batches = tree.iterate(
                step_size=step_size,
                filter_branch=lambda TBranch: TBranch.name in branch_names,
                aliases=aliases,
                entry_start=start,
                entry_stop=stop,
                library="pd",
            )
        else:
            batches = (
                __apply_cut_and_convert_batch(batch, cut, aliases)
                for batch in tree.iterate(
                    step_size=step_size,
                    filter_branch=lambda TBranch: TBranch.name in branch_names,
                    entry_start=start,
                    entry_stop=stop,
                    library="ak",
                )
            )

        for batch in batches:
            if observable_renames:
                batch.rename(columns=observable_renames, inplace=True)
            data_sets.append(__expand_awkward_cols(batch))

    collected_data = pd.concat(data_sets).reset_index(level=0, drop=True)
    return DataSet(
        collected_data,
        observable_names=collected_data.columns,
    )


def __apply_cut_and_convert_batch(
    batch: ak.Array,
    cut: str,
    aliases: Optional[Dict[str, str]],
) -> pd.DataFrame:
    """Apply an event cut to an Awkward batch and return its pandas form."""
    values = {field: batch[field] for field in batch.fields}
    for name, expression in (aliases or {}).items():
        values[name] = eval(expression, {"__builtins__": {}}, values)

    mask = eval(cut, {"__builtins__": {}}, values)
    if not isinstance(mask, ak.Array):
        mask = ak.Array(mask)
    if ak.to_layout(mask).purelist_depth > 1:
        # Empty jagged collections must not turn a particle predicate into
        # True through ``all([])`` when it is combined with scalar predicates.
        mask = ak.all(mask, axis=1) & ak.any(mask, axis=1)

    selected = batch[mask]
    columns = {}
    for field in selected.fields:
        values = selected[field]
        if ak.to_layout(values).purelist_depth > 1:
            columns[field] = pd.Series(values.tolist(), dtype="awkward")
        else:
            columns[field] = np.asarray(values)
    return pd.DataFrame(columns)


def __expand_awkward_cols(
    batch: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand awkward columns in a DataFrame to multiple columns.
    """
    expanded_batch = batch.copy()
    
    # Split awkward cols to multiple cols
    for col in batch.columns:
        # Check if the column contains awkward arrays
        if len(batch[col]) > 0 and isinstance(batch[col].iloc[0], ak.Array):
            split_col = pd.DataFrame(batch[col].to_list())
            split_col.index = expanded_batch.index
            col_position: int = expanded_batch.columns.get_loc(col)
            expanded_batch.drop(col, axis=1, inplace=True)
            
            for i in range(split_col.shape[1]):
                expanded_batch.insert(col_position + i, f"{col}_{i}", split_col[i])
    
    return expanded_batch

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
