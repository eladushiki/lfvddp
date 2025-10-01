from data_tools.data_utils import DataSet
import numpy as np


def load_numpy_events(path, number_of_events=None) -> DataSet:
    event_data = np.load(path)
    
    if number_of_events is None:
        loaded_dataset = DataSet(event_data)
    else:
        loaded_dataset = DataSet(event_data[:number_of_events])

    return loaded_dataset
