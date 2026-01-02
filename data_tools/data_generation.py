from typing import Dict, Tuple
from data_tools.data_utils import DataSet, resample as ddp_resample
from data_tools.dataset_config import DatasetConfig, DatasetParameters, GeneratedDatasetParameters, LoadedDatasetParameters
from frame.context.execution_context import ExecutionContext


class DataGeneration:

    _instance = None
    _loaded_datasets: Dict[str, Tuple[DataSet, DataSet]] = {}

    def __new__(cls, context: ExecutionContext):
        if cls._instance is None:
            cls._instance = super(DataGeneration, cls).__new__(cls)
            cls._instance.__init__(context)
        return cls._instance

    def __init__(self, context: ExecutionContext):
        self._context = context
        self._config: DatasetConfig = context.config

    def __getitem__(self, item: str) -> Tuple[DataSet, DatasetParameters]:
        try:
            dataset_parameters = self._config.get_parameters(item)
            dataset = self.__retrieve_dataset(dataset_parameters, name=item)
            return dataset, dataset_parameters

        except KeyError:
            raise KeyError(f"Dataset '{item}' not found in the configuration.")

    def __retrieve_dataset(self, dataset_parameters: DatasetParameters, name: str) -> DataSet:
        """
        Implements loading, generation and resampling of different datasets while holding global
        state for them.
        Signal and background numbers of events are kept as specified and are resampled
        separately if needed.
        """
        # In case of a generated dataset, just generate the data
        if isinstance(dataset_parameters, GeneratedDatasetParameters):
            background_data, signal_data = dataset_parameters.dataset__data
            
        # In case of a loaded dataset, we keep track of the remaining data to enable resampling mechanism
        elif isinstance(dataset_parameters, LoadedDatasetParameters):
            try:
                background_data, signal_data = self._loaded_datasets[dataset_parameters.name]
            except KeyError:
                background_data, signal_data = dataset_parameters.dataset__data
                self._loaded_datasets[dataset_parameters.name] = (background_data, signal_data)
            
            if background_data.n_samples < dataset_parameters.dataset__number_of_background_events:
                raise ValueError(f"Loaded dataset {dataset_parameters.name} has only {background_data.n_samples} "\
                    f"samples left, but requested {dataset_parameters.dataset__number_of_background_events} samples.")
            
            if dataset_parameters.dataset_loaded__resample_is_resample:
                background_data, background_remainder = ddp_resample(
                    background_data,
                    dataset_parameters.dataset__number_of_background_events,
                    replacement=dataset_parameters.dataset_loaded__resample_is_replacement,
                )
                signal_data, signal_remainder = ddp_resample(
                    signal_data,
                    dataset_parameters.dataset__number_of_signal_events,
                    replacement=dataset_parameters.dataset_loaded__resample_is_replacement,
                )
                self._loaded_datasets[dataset_parameters.name] = (background_remainder, signal_remainder)
            
        else:
            raise ValueError(f"Unsupported dataset parameters type: {type(dataset_parameters)}")

        return background_data + signal_data
