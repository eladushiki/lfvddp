from logging import debug, info
import numpy as np
from pathlib import Path
from h5py import File
import tensorflow as tf

def save_training_history(model_history: tf.keras.callbacks.History, weights_file_path: Path, epochs: int, epochs_checkpoint: int = 1):
    with File(weights_file_path,"w") as history_file:
        epoch       = np.array(range(epochs))
        patience_t  = epochs_checkpoint
        keepEpoch   = epoch % patience_t == 0
        history_file.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
        for key in list(model_history.history.keys()):
            monitored = np.array(model_history.history[key])
            debug('%s: %f'%(key, monitored[-1]))
            history_file.create_dataset(key, data=monitored[keepEpoch], compression='gzip')
        info("saved history")


def load_training_history(weights_filename: Path):
    with File(weights_filename, "r") as history_file:
        keys = [(key) for key in list (history_file.keys())]
        history = history_file.get(str(keys[2]))  # which is, "loss"
        history = np.array(history)
        return history
