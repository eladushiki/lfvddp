from typing import Dict, Iterable, Iterator, Tuple
from data_tools.data_utils import (
    DATASET_REGIONS,
    DataSet,
    DataSetRegion,
    ShiftAndNormalizationFactor,
    resample as ddp_resample,
)
from data_tools.dataset_config import DatasetConfig, DatasetParameters, GeneratedDatasetParameters, LoadedDatasetParameters
from data_tools.dataset_pair import RegionalDataPair
from frame.context.execution_context import ExecutionContext

class DataBatch:
    """
    All the data sets needed for a single training run.
    """
    REQUIRED_DATASET_CATEGORIES = [
        DATASET_REGIONS.sr.a,
        DATASET_REGIONS.cr.a,
        DATASET_REGIONS.sr.b,
        DATASET_REGIONS.cr.b,
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
        sr = self.__retrieve_regional_pair(DATASET_REGIONS.sr)
        cr = self.__retrieve_regional_pair(DATASET_REGIONS.cr)
        return DataBatch(
            [
                (sr.a, sr.a_parameters),
                (cr.a, cr.a_parameters),
                (sr.b, sr.b_parameters),
                (cr.b, cr.b_parameters),
            ]
        )

    def __getitem__(self, item: DataSet.DataSetCategory) -> Tuple[DataSet, DatasetParameters]:
        try:
            region = DATASET_REGIONS.for_category(item)
            if self.__has_parameters(region.a) and self.__has_parameters(region.b):
                return self.__retrieve_regional_pair(region).get(item)

            dataset_parameters = self._config.get_parameters(item)
            return self.__materialize_dataset(dataset_parameters), dataset_parameters

        except KeyError:
            raise KeyError(f"Dataset category '{item}' not found in the configuration.")

    def __has_parameters(self, category: DataSet.DataSetCategory) -> bool:
        try:
            self._config.get_parameters(category)
        except KeyError:
            return False
        return True

    def __retrieve_regional_pair(
        self,
        region: DataSetRegion,
    ) -> RegionalDataPair:
        a_parameters = self._config.get_parameters(region.a)
        b_parameters = self._config.get_parameters(region.b)
        return RegionalDataPair(
            a=self.__materialize_dataset(a_parameters),
            a_parameters=a_parameters,
            b=self.__materialize_dataset(b_parameters),
            b_parameters=b_parameters,
            split_policy=a_parameters.dataset__regional_split_policy,
        ).finalized()

    def __materialize_dataset(self, dataset_parameters: DatasetParameters) -> DataSet:
        """
        Materialize one category before its regional A/B finalization.

        Signal and background numbers of events are kept as specified and are
        resampled separately when configured for a loaded dataset.
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
