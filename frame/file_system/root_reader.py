from logging import error, info
from pathlib import Path
from typing import Any, Dict, List
import uproot


def load_root_events(
    XRootD_url: str,
    tree_key: str = "Events",
    branch_names: List[str] = [],
    start: int = 0,
    stop: int = 1000,
    library: str = "np",
):
    try:
        with uproot.open(XRootD_url) as file:
            tree = file[tree_key]

            arrays = tree.arrays(
                branch_names,
                entry_start=start,
                entry_stop=stop,
                library=library,
            )
            info(f"Read items from {tree_key} numbers {start} to {stop} of {tree.num_entries}")

            return arrays
        
    except KeyError:
        error(f"KeyError: The key '{tree_key}' does not exist in the ROOT file at {XRootD_url}.")


def save_root_events(
    root_file_path: Path,
    tree_key: str = "Events",
    branches: Dict[str, Any] = {},
):
    with uproot.recreate(root_file_path) as root_file:
        root_file[tree_key] = branches

        info(f"Saved branches {list(branches.keys())} to {root_file_path} root file.")
