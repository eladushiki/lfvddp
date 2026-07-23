from glob import glob
from logging import warning
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from data_tools.dataset_config import DatasetConfig, DatasetParameters
from data_tools.detector.detector_config import DetectorConfig
from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_continuous,
)
from frame.context.execution_context import ExecutionContext
from frame.context.execution_products import unstamp_product_stem
from frame.file_structure import (
    RESULTING_T_FILE_STEM,
    TRAINING_HISTORY_LOG_FILE_SUFFIX,
    TRAINING_RESULT_FILE_EXTENSION,
)
from frame.file_system.training_history import HistoryKeys, load_training_history
from train.train_config import TrainConfig


def utils__get_signal_dataset_parameters(
        signal_context: ExecutionContext,
) -> DatasetParameters:

    # Validate signal configuration
    signal_dataset_parameters = None

    signal_config: Union[DatasetConfig, TrainConfig] = signal_context.config
    for dataset_parameters in signal_config.dataset_parameters:

        # We do validate that there is a signal in at most one dataset
        # In low signal counts, de-facto number of signal events might vanish. If so, check intentions
        # by looking at mean.
        if dataset_parameters.dataset__has_signal:
            assert signal_dataset_parameters is None, \
                f"multiple signal datasets found, {dataset_parameters.category} being the second"

            signal_dataset_parameters = dataset_parameters

    if not signal_dataset_parameters:
        raise ValueError("No signal dataset found in the configuration")

    return signal_dataset_parameters


class ResultAggregator:
    def __init__(self, parent_directory: Path):
        self._parent_directory = parent_directory
        if not self._parent_directory.is_dir():
            raise NotADirectoryError(f"Parent directory {self._parent_directory} does not exist")

        # Exhibits retrieved
        self._test_statistics = None
        self._history_values = None
        self._epochs = None
        self._run_contexts = None

        # Load t-values
        self._load_t_values()

    def _load_t_values(self):
        # Find all files
        _files_in_output_dir = glob(str(self._parent_directory) + f"/**/{RESULTING_T_FILE_STEM}*.{TRAINING_RESULT_FILE_EXTENSION}", recursive=True)

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
        
        required_value_keys = (
            HistoryKeys.NUMERATOR.value,
            HistoryKeys.DENOMINATOR.value,
            HistoryKeys.T.value,
        )
        loaded = []
        for history_file in all_history_files:
            history = load_training_history(Path(history_file))
            missing_keys = {
                HistoryKeys.EPOCH.value,
                *required_value_keys,
            } - history.keys()
            if missing_keys:
                raise ValueError(
                    f"History {history_file} is not a paired t history; "
                    f"missing {sorted(missing_keys)}"
                )
            sample_name = Path(unstamp_product_stem(Path(history_file))).stem
            run_output = str(Path(history_file).parent.parent)
            loaded.append((run_output, sample_name, history_file, history))

        epochs = np.asarray(loaded[0][3][HistoryKeys.EPOCH.value])
        for _, _, history_file, history in loaded[1:]:
            if not np.array_equal(epochs, history[HistoryKeys.EPOCH.value]):
                raise ValueError(f"Epochs in {history_file} are not aligned.")

        run_outputs = sorted({item[0] for item in loaded})
        sample_names = sorted({item[1] for item in loaded})
        history_values = {
            sample_name: {
                key: np.full((len(run_outputs), len(epochs)), np.nan)
                for key in required_value_keys
            }
            for sample_name in sample_names
        }

        seen = set()
        for run_output, sample_name, _, history in loaded:
            identity = (run_output, sample_name)
            if identity in seen:
                raise ValueError(
                    f"Found multiple {sample_name} histories in {run_output}."
                )
            seen.add(identity)
            run_index = run_outputs.index(run_output)
            for key in required_value_keys:
                history_values[sample_name][key][run_index] = history[key]

        self._history_values = history_values
        self._test_statistics = np.sum(
            [values[HistoryKeys.T.value] for values in history_values.values()],
            axis=0,
        )
        self._epochs = epochs

    @property
    def all_test_statistics(self) -> NDArray[np.float64]:
        if self._test_statistics is None:
            self._load_test_statistics()
        return self._test_statistics

    @property
    def all_history_values(self) -> dict[str, dict[str, NDArray[np.float64]]]:
        """Numerator, denominator, and t histories grouped by sample name."""
        if self._history_values is None:
            self._load_test_statistics()
        return self._history_values

    @property
    def all_epochs(self) -> NDArray[np.int64]:
        if self._epochs is None:
            self._load_test_statistics()
        return self._epochs

    def _load_run_contexts(self):
        self._run_contexts = [
            context
            for context, _ in ExecutionContext.discover_run_contexts(self._parent_directory)
        ]

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
