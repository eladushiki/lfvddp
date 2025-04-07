from typing import Dict
from data_tools.data_utils import DataSet, DetectorEffect
from data_tools.dataset_config import DatasetConfig, DatasetParameters, GeneratedDatasetParameters
from data_tools.event_generation.exp import exp
from data_tools.event_generation.gauss import gauss
from data_tools.event_generation.physics import physics


class DataGeneration:

    _instance = None

    GeneratedDatasetTypes = {
        'exp': exp,
        'gauss': gauss,
        'physics': physics,
    }

    def __new__(cls, config: DatasetConfig):
        if cls._instance is None:
            cls._instance = super(DataGeneration, cls).__new__(cls)
            cls._instance.__init__(config)
        return cls._instance

    def __init__(self, config: DatasetConfig):
        self._config = config
        self._datasets: Dict[str, DataSet] = {}

    def __getitem__(self, item: str) -> DataSet:
        # Lazily create datasets
        try:
            return self._datasets[item]
        except KeyError:
            pass 

        try:
            dataset_parameters = self._config.get_parameters(item)
            self._datasets[item] = self.__create_dataset(dataset_parameters)
            return self._datasets[item]
        except KeyError:
            raise KeyError(f"Dataset '{item}' not found in the configuration.")

    def __create_dataset(self, dataset_parameters: DatasetParameters) -> DataSet:
        if isinstance(dataset_parameters, GeneratedDatasetParameters):
            generating_function = self.GeneratedDatasetTypes[dataset_parameters.dataset__background_data_generation_function]
            data = generating_function(dataset_parameters, **dataset_parameters.dataset__function_specific_additional_parameters)

            detector = DetectorEffect(
                dataset_parameters.dataset__detector_efficiency,
                dataset_parameters.dataset__detector_error
            )
            data.apply_detector_effect(detector)

            if dataset_parameters.dataset__resample_is_resample:
                raise NotImplementedError("Resampling is not implemented")
        else:
            raise ValueError(f"Unsupported dataset parameters type: {type(dataset_parameters)}")
        return data
