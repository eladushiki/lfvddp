import h5py
from logging import debug, info
import numpy as np
from pathlib import Path
import tensorflow as tf

def save_training_history(model_history: tf.keras.callbacks.History, weights_file_path: Path, epochs: int, epochs_checkpoint: int = 1):
    with h5py.File(weights_file_path,"w") as history_file:
        epoch       = np.array(range(epochs))
        patience_t  = epochs_checkpoint
        keepEpoch   = epoch % patience_t == 0
        history_file.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
        for key in list(model_history.history.keys()):
            monitored = np.array(model_history.history[key])
            debug('%s: %f'%(key, monitored[-1]))
            history_file.create_dataset(key, data=monitored[keepEpoch], compression='gzip')
        info("saved history")


def load_training_history(file_path: Path):
    with h5py.File(file_path, 'r') as f:
        history = {key: f[key][()] for key in f.keys()}
    return history
