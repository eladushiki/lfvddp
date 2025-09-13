from copy import deepcopy
from typing import Dict, Union
from data_tools.data_utils import DataSet, DetectorEffect, resample as ddp_resample
from data_tools.dataset_config import DatasetConfig, DatasetParameters, GeneratedDatasetParameters, LoadedDatasetParameters
from data_tools.detector.detector_config import DetectorConfig
from frame.context.execution_context import ExecutionContext
from plot.plots import plot_data_generation_sliced


class DataGeneration:

    _instance = None
    _loaded_datasets: Dict[str, DataSet] = {}

    def __new__(cls, context: ExecutionContext):
        if cls._instance is None:
            cls._instance = super(DataGeneration, cls).__new__(cls)
            cls._instance.__init__(context)
        return cls._instance

    def __init__(self, context: ExecutionContext):
        self._context = context
        self._config: Union[
            DatasetConfig,
            DetectorConfig,
        ] = context.config
        self._datasets: Dict[str, DataSet] = {}

    def __getitem__(self, item: str) -> DataSet:
        # Lazily create datasets
        try:
            return self._datasets[item]
        except KeyError:
            pass 

        try:
            dataset_parameters = self._config.get_parameters(item)
            self._datasets[item] = self.__create_dataset(dataset_parameters, name=item)
            return self._datasets[item]

        except KeyError:
            raise KeyError(f"Dataset '{item}' not found in the configuration.")

    def __create_dataset(self, dataset_parameters: DatasetParameters, name: str) -> DataSet:
        """
        Implements mechanisms of different datasets while holding global state for them.
        """
        # In case of a generated dataset, just generate the data
        if isinstance(dataset_parameters, GeneratedDatasetParameters):
            loaded_data = dataset_parameters.dataset__data
            
        # In case of a loaded dataset, we keep track of the remaining data to enable resampling mechanism
        elif isinstance(dataset_parameters, LoadedDatasetParameters):
            try:
                loaded_data = self._loaded_datasets[dataset_parameters.dataset_loaded__file_name]
            except KeyError:
                loaded_data = dataset_parameters.dataset__data
            
            if loaded_data.n_samples < dataset_parameters.dataset__number_of_background_events:
                raise ValueError(f"Loaded dataset has only {loaded_data.n_samples} samples, "\
                    f"but requested {dataset_parameters.dataset__number_of_background_events} samples.")
            
            if dataset_parameters.dataset_loaded__resample_is_resample:
                loaded_data, self._loaded_datasets[dataset_parameters.dataset_loaded__file_name] = ddp_resample(
                    loaded_data,
                    dataset_parameters.dataset__number_of_background_events,
                    replacement=dataset_parameters.dataset_loaded__resample_is_replacement,
                )
            else:
                self._loaded_datasets[dataset_parameters.dataset_loaded__file_name] = loaded_data
            
        else:
            raise ValueError(f"Unsupported dataset parameters type: {type(dataset_parameters)}")

        original_data = deepcopy(loaded_data)

        detector = DetectorEffect(
            efficiency_function=dataset_parameters.dataset__detector_efficiency,
            observable_names=self._config.detector__detect_observable_names,
            binning_minima=self._config.detector__binning_minima,
            binning_maxima=self._config.detector__binning_maxima,
            numbers_of_bins=self._config.detector__binning_number_of_bins,
            efficiency_uncertainty_function=dataset_parameters.dataset__detector_efficiency_uncertainty,
            error_function=dataset_parameters.dataset__detector_error
        )
        loaded_data = detector.affect_and_compensate(loaded_data)

        if self._context.is_debug_mode:
            figure = plot_data_generation_sliced(
                context=self._context,
                original_sample=original_data,
                processed_sample=loaded_data,
                bins=detector._dimensional_bin_centers[self._config.detector__detect_observable_names[0]],
                xlabel=f"{self._config.detector__detect_observable_names[0]}",
            )
            self._context.save_and_document_figure(figure, self._context.unique_out_dir / f"{name}_data_process_plot.png")

        return loaded_data
