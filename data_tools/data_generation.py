import re
from typing import Dict, Iterable, Iterator, Tuple
from data_tools.data_utils import DataSet, ShiftAndNormalizationFactor, resample as ddp_resample
from data_tools.dataset_config import DatasetConfig, DatasetParameters, GeneratedDatasetParameters, LoadedDatasetParameters
from frame.context.execution_context import ExecutionContext

class DataBatch:
    """
    All the data sets needed for a single training run.
    """
    REQUIRED_DATASET_CATEGORIES = [
        DataSet.DataSetCategory.A_SR,
        DataSet.DataSetCategory.A_CR,
        DataSet.DataSetCategory.B_SR,
        DataSet.DataSetCategory.B_CR,
    ]

    def __init__(self, dss_and_params: Iterable[Tuple[DataSet, DatasetParameters]]):
        self.datasets: Dict[DataSet.DataSetCategory, DataSet] = {}
        self.parameters: Dict[DataSet.DataSetCategory, DatasetParameters] = {}
        for ds, params in dss_and_params:
            if (category := ds.category) not in self.REQUIRED_DATASET_CATEGORIES:
                raise ValueError(f"Dataset category {category} is not required for training.")
            if category in self.datasets.keys():
                raise ValueError(f"Duplicate dataset for category {category}.")

            self.datasets[category], self.parameters[category] = ds, params

        for cat in DataBatch.REQUIRED_DATASET_CATEGORIES:
            if cat not in self.datasets.keys():
                raise ValueError(f"Missing dataset for required category {cat}.")

    def __iter__(self) -> Iterator[Tuple[DataSet, DatasetParameters]]:
        for cat in DataBatch.REQUIRED_DATASET_CATEGORIES:
            yield self.datasets[cat], self.parameters[cat]

    def swap_ab(self):
        """
        Swap the A and B datasets in the batch.
        """
        self.datasets[DataSet.DataSetCategory.A_SR], self.datasets[DataSet.DataSetCategory.B_SR] = \
            self.datasets[DataSet.DataSetCategory.B_SR], self.datasets[DataSet.DataSetCategory.A_SR]
        self.datasets[DataSet.DataSetCategory.A_SR].category = DataSet.DataSetCategory.A_SR
        self.datasets[DataSet.DataSetCategory.B_SR].category = DataSet.DataSetCategory.B_SR
        self.datasets[DataSet.DataSetCategory.A_CR], self.datasets[DataSet.DataSetCategory.B_CR] = \
            self.datasets[DataSet.DataSetCategory.B_CR], self.datasets[DataSet.DataSetCategory.A_CR]
        self.datasets[DataSet.DataSetCategory.A_CR].category = DataSet.DataSetCategory.A_CR
        self.datasets[DataSet.DataSetCategory.B_CR].category = DataSet.DataSetCategory.B_CR

    @property
    def unified_data(self) -> DataSet:
        return sum((dataset for dataset, _ in self), DataSet())

    def get_normalized(self) -> Tuple['DataBatch', ShiftAndNormalizationFactor]:
        _, norm_factor = self.unified_data.get_normalized()
        return DataBatch([
            (ds / norm_factor, params) for ds, params in self
        ]), norm_factor

class DataGeneration:

    _instance = None
    _loaded_datasets: Dict[DataSet.DataSetCategory, Tuple[DataSet, DataSet]] = {}

    def __new__(cls, context: ExecutionContext):
        if cls._instance is None:
            cls._instance = super(DataGeneration, cls).__new__(cls)
            cls._instance.__init__(context)
        return cls._instance

    def __init__(self, context: ExecutionContext):
        self._context = context
        self._config: DatasetConfig = context.config

    def get_batch(self) -> DataBatch:
        return DataBatch(
            [self[category] for category in DataBatch.REQUIRED_DATASET_CATEGORIES]
        )

    def __getitem__(self, item: DataSet.DataSetCategory) -> Tuple[DataSet, DatasetParameters]:
        try:
            dataset_parameters = self._config.get_parameters(item)
            dataset = self.__retrieve_dataset(dataset_parameters)
            return dataset, dataset_parameters

        except KeyError:
            raise KeyError(f"Dataset category '{item}' not found in the configuration.")

    def __retrieve_dataset(self, dataset_parameters: DatasetParameters) -> DataSet:
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
                background_data, signal_data = self._loaded_datasets[dataset_parameters.category]
            except KeyError:
                background_data, signal_data = dataset_parameters.dataset__data
                self._loaded_datasets[dataset_parameters.category] = (background_data, signal_data)
            
            if background_data.n_samples < dataset_parameters.dataset__number_of_background_events:
                raise ValueError(f"Loaded dataset of category {dataset_parameters.category} has only {background_data.n_samples} "\
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
                self._loaded_datasets[dataset_parameters.category] = (background_remainder, signal_remainder)
            
        else:
            raise ValueError(f"Unsupported dataset parameters type: {type(dataset_parameters)}")

        complete_ds = background_data + signal_data
        complete_ds.category = dataset_parameters.category
        return complete_ds
