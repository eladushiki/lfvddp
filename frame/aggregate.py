from glob import glob
from logging import warning
from pathlib import Path
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_structure import SINGLE_TRAINING_RESULT_FILE_EXTENSION, TRAINING_HISTORY_FILE_EXTENSION
from frame.file_system.training_history import HistoryKeys, load_training_history
import numpy as np
from numpy._typing._generic_alias import NDArray
from plot.plotting_config import PlottingConfig
from train.train_config import TrainConfig


class ResultAggregator:
    def __init__(self, context: ExecutionContext):
        self._context = context

        if not isinstance((config := context.config), PlottingConfig):
            raise TypeError(f"Expected PlottingConfig, got {type(config).__name__}")
        if not isinstance(config, TrainConfig):
            raise ValueError(f"Expected TrainConfig, got {type(config).__name__}")
    
        self._config = config

        self._train_output_directory = Path(self._config.plot__target_run_parent_directory)
        if not self._train_output_directory.is_dir():
            raise NotADirectoryError(f"Parent directory {self._train_output_directory} does not exist")

        # Exhibits retrieved
        self._t_values = None
        self._test_statistics = None
        self._epochs = None

    def _load_t_values(self):
        # Find all files
        _files_in_output_dir = glob(str(self._train_output_directory) + f"/**/*.{SINGLE_TRAINING_RESULT_FILE_EXTENSION}", recursive=True)

        # Read and validate content
        aggregated_results = []
        for _file in _files_in_output_dir:
            try:
                with open(_file, 'r') as f:
                    _content = f.read()
                _result = (float(_content), _file)
            except ValueError:
                warning(f"Could not parse training result from file {_file}")
                continue
            aggregated_results.append(_result)

        self._t_values = aggregated_results

    @property
    def all_t_values(self) -> NDArray[np.float64]:
        if self._t_values is None:
            self._load_t_values()
        return np.array([t[0] for t in self._t_values])
    
    def _load_test_statistics(self):
        # Gather history files
        all_history_files = glob(str(self._config.plot__target_run_parent_directory) + f"/**/*.{TRAINING_HISTORY_FILE_EXTENSION}", recursive=True)
        if not all_history_files:
            raise ValueError("No history files found")
        
        # Load history files
        all_loaded_histories = {history_file: load_training_history(Path(history_file)) for history_file in all_history_files}
        
        # Epochs should be aligned in all files. If you get a 1D array here, they're not of the same length.
        all_epochs = np.array([history[HistoryKeys.EPOCH.value] for history in all_loaded_histories.values()])
        for col in range(all_epochs.shape[1]):
            if not (m := np.maximum.reduce(all_epochs[:, col], initial=0)) == np.minimum.reduce(all_epochs[:, col], initial=m):
                raise ValueError("Epochs are not the same for all files")
        epochs = all_epochs[0]

        # We assume each run generates each type of test statistic, and that they all should be summed to get a single value
        unique_history_file_names = np.unique([Path(history_file).name for history_file in all_history_files])
        unique_runs_output_dirs = np.unique([str(Path(history_file).parent) for history_file in all_history_files])

        # We assume that we need to sum two types of test statistics for every single value, each with different name
        all_model_t_test_statistics = np.zeros(shape=(len(unique_runs_output_dirs), len(epochs)))
        for run_index, run_output in enumerate(unique_runs_output_dirs):
            for history_file_name in unique_history_file_names:
                history = all_loaded_histories[run_output + '/' + history_file_name]
                all_model_t_test_statistics[run_index, :] += np.array(calc_t_test_statistic(history[HistoryKeys.LOSS.value]))  # type: ignore

        self._test_statistics = all_model_t_test_statistics
        self._epochs = epochs

    @property
    def all_test_statistics(self) -> NDArray[np.float64]:
        if self._test_statistics is None:
            self._load_test_statistics()
        return self._test_statistics

    @property
    def all_epochs(self) -> NDArray[np.int64]:
        if self._epochs is None:
            self._load_test_statistics()
        return self._epochs
