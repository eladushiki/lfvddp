from glob import glob
from logging import warning
from pathlib import Path
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import calc_injected_t_significance_by_sqrt_q0_continuous, calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.context.execution_products import products_from_stem, unstamp_product_stem
from frame.file_structure import CONTEXT_FILE_NAME, SINGLE_TRAINING_RESULT_FILE_EXTENSION, TRAINING_HISTORY_LOG_FILE_SUFFIX
from frame.file_system.training_history import HistoryKeys, load_training_history
import numpy as np
from numpy.typing import NDArray
from plot.plot_utils import utils__get_signal_dataset_parameters


class ResultAggregator:
    def __init__(self, parent_directory: Path):
        self._parent_directory = parent_directory
        if not self._parent_directory.is_dir():
            raise NotADirectoryError(f"Parent directory {self._parent_directory} does not exist")

        # Exhibits retrieved
        self._test_statistics = None
        self._epochs = None
        self._run_contexts = None

        # Load t-values
        self._load_t_values()

    def _load_t_values(self):
        # Find all files
        _files_in_output_dir = glob(str(self._parent_directory) + f"/**/*.{SINGLE_TRAINING_RESULT_FILE_EXTENSION}", recursive=True)

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
        return np.array([t[0] for t in self._t_values if not np.isnan(t[0])]) # type: ignore
    
    @property
    def nan_t_values(self) -> int:
        return len([t[0] for t in self._t_values if np.isnan(t[0])])

    def _load_test_statistics(self):
        # Gather history files
        all_history_files = glob(str(self._parent_directory) + f"/**/*.{TRAINING_HISTORY_LOG_FILE_SUFFIX}", recursive=True)
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
        unique_history_file_stems = np.unique([unstamp_product_stem(Path(history_file)) for history_file in all_history_files])
        unique_runs_output_dirs = np.unique([str(Path(history_file).parent) for history_file in all_history_files])

        # We assume that we need to sum two types of test statistics for every single value, each with different name
        all_model_t_test_statistics = np.zeros(shape=(len(unique_runs_output_dirs), len(epochs)))
        for run_index, run_output in enumerate(unique_runs_output_dirs):
            for history_file_stem in unique_history_file_stems:    
                history_files = [f for f in products_from_stem(history_file_stem, Path(run_output)) if str(f) in all_history_files]
                
                if len(history_files) != 1:
                    raise ValueError(f"Found multiple history files for stem {history_file_stem} in directory {run_output}")
                
                history_file = all_loaded_histories[str(history_files[0])]
                all_model_t_test_statistics[run_index, :] += np.array(calc_t_test_statistic(history_file[HistoryKeys.LOSS.value]))  # type: ignore

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

    def _load_run_contexts(self):
        _context_files = glob(str(self._parent_directory) + f"**/{CONTEXT_FILE_NAME}", recursive=True)
        self._run_contexts = [ExecutionContext.naive_load_from_file(Path(_context_file)) for _context_file in _context_files]

    @property
    def all_injected_significances(self) -> NDArray[np.float64]:
        if self._test_statistics is None:
            self._load_run_contexts()

        injected_significances = []
        for context in self._run_contexts:
            signal_dataset_parameters = utils__get_signal_dataset_parameters(context)
            detector_config: DetectorConfig = context.config
            injected_significances.append(calc_injected_t_significance_by_sqrt_q0_continuous(
                background_pdf=signal_dataset_parameters.dataset_generated__background_pdf,
                signal_pdf=signal_dataset_parameters.dataset_generated__signal_pdf,
                n_background_events=signal_dataset_parameters.dataset__number_of_background_events,
                n_signal_events=signal_dataset_parameters.dataset__number_of_signal_events,
                upper_limit=detector_config.detector__binning_maxima[0], # ohhh this is going to break at dim>=2
            ))

        return np.array(injected_significances)
