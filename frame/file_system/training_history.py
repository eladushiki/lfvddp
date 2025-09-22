from enum import Enum
from typing import Any, Dict
import h5py
from logging import debug, info
import numpy as np
from pathlib import Path


class HistoryKeys(Enum):
    LOSS = "loss"
    PREDICTION_LOSS = "prediction_loss"
    NUISANCE_LOSS = "nuisance_loss"
    SINGLE_NUISANCE_LOSS = "single_nuisance_loss"
    NUISANCE_ABS_SUM = "nuisance_abs_sum"
    EPOCH = "epoch"


def save_training_history(model_history: Dict[str, Any], history_path: Path, epochs: int, epochs_checkpoint: int = 1):    
    with h5py.File(history_path,"w") as history_file:
        for key in list(model_history.keys()):
            monitored = np.array(model_history[key])
            debug('%s: %f'%(key, monitored[-1]))
            history_file.create_dataset(key, data=monitored, compression='gzip')
        info("saved history")


def load_training_history(file_path: Path):
    with h5py.File(file_path, 'r') as f:
        history = {key: f[key][()] for key in f.keys()}
    return history
