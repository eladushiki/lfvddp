from pathlib import Path
import h5py

def load_training_history(file_path: Path):
    with h5py.File(file_path, 'r') as f:
        history = {key: f[key][()] for key in f.keys()}
    return history


# todo: move here save_training_history whatever it's called
