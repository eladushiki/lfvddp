import awkward as ak
import numpy as np
import uproot

from frame.file_system.root_reader import load_root_events


def test_load_root_events_reduces_jagged_cut_to_event_mask(tmp_path):
    root_path = tmp_path / "events.root"
    with uproot.recreate(root_path) as root_file:
        root_file["Events"] = {
            "nTau": [1, 1, 1],
            "nMuon": [0, 0, 0],
            "nElectron": [1, 0, 2],
            "Electron_pt": ak.Array([[10.0], [], [6000.0, 20.0]]),
        }

    events = load_root_events(
        root_path,
        branch_names=["nTau", "nMuon", "nElectron", "Electron_pt"],
        cut="(nTau == 1) & (nMuon == 0) & (nElectron == 1) & (Electron_pt < 5000)",
    ).events

    np.testing.assert_array_equal(events[:, 0], [10.0])
    np.testing.assert_array_equal(events[:, 1:], [[1, 0, 1]])
