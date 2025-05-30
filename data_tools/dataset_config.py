from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Type, Union

from camel_converter import to_pascal

from data_tools.data_utils import DataSet
from data_tools.event_generation import background, signal
from data_tools.event_generation.distribution import DataDistribution
from data_tools.event_generation.types import FLOAT_OR_ARRAY
from frame.module_retriever import _retrieve_from_module
import numpy as np
from os.path import isfile


@dataclass
class DatasetParameters(ABC):
    
    _dataset__number_of_dimensions: int = field(init=False)
    # For documentation purposes
    name: str
    type: str

    # Background parameters
    dataset__mean_number_of_background_events: int

    # Signal parameters
    dataset__signal_data_generation_function: str = field(default="")
    dataset__mean_number_of_signal_events: int = field(default=0)
    dataset__signal_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Detector simulation
    dataset__detector_efficiency: str = field(default="")
    dataset__detector_efficiency_uncertainty: str = field(default="")
    dataset__detector_error: str = field(default="")

    # Induced nuisance parameters
    dataset__induced_shape_nuisance_value: float = field(default=0.0)
    dataset__induced_norm_nuisance_value: float = field(default=0.0)

    # Created automatically
    ## Picked poissonically based on mean numbers
    dataset__number_of_signal_events: int = field(default=None)
    dataset__number_of_background_events: int = field(default=None)


    @classmethod
    @abstractmethod
    def DATASET_PARAMTER_TYPE_NAME(cls) -> str:
        pass

    @property
    @abstractmethod
    def dataset__data(self) -> DataSet:
        pass

    def __post_init__(self):
        if not self.dataset__number_of_background_events:
            self.dataset__number_of_background_events = np.random.poisson(
                lam=self.dataset__mean_number_of_background_events * np.exp(self.dataset__induced_norm_nuisance_value),
                size=1,
            ).item() if self.dataset__mean_number_of_background_events > 0 else 0
        
        if not self.dataset__number_of_signal_events:
            self.dataset__number_of_signal_events = np.random.poisson(
                lam=self.dataset__mean_number_of_signal_events * np.exp(self.dataset__induced_norm_nuisance_value),
                size=1,
            ).item() if self.dataset__mean_number_of_signal_events > 0 else 0

    
@dataclass
class LoadedDatasetParameters(DatasetParameters):
        
    @classmethod
    def DATASET_PARAMTER_TYPE_NAME(cls) -> str:
        return "loaded"

    def __post_init__(self):
        super().__post_init__()

        # Make sure the file exists
        if not isfile(self.dataset__loaded_file_name):
            raise FileNotFoundError(f"Loaded file '{self.dataset__loaded_file_name}' does not exist.")

    # Data source
    dataset__loaded_file_name: str = field(default="")

    # Resampling settings
    dataset__resample_is_resample: bool = field(default=False)
    dataset__resample_is_replacement: bool = field(default=False)

    @property
    def dataset__data(self) -> DataSet:
        """
        Load the data from the specified file, and update the internal
        state of loaded data to match resampling settings.
        """
        data = np.load(self.dataset__loaded_file_name)
        return DataSet(data)


@dataclass
class GeneratedDatasetParameters(DatasetParameters, ABC):

    @classmethod
    def DATASET_PARAMTER_TYPE_NAME(cls) -> str:
        return "generated"
    
    # Additional background parameters
    # This is the defining attribute for the subclass
    dataset_generated__background_function: str = field(default="")
    dataset_generated__background_parameters: Dict[str, Any] = field(default_factory=dict)

    @property
    def __dataset_generated__background_distribution(self) -> Union[str, DataDistribution]:
        """
        Get the background PDF function based on the configuration.
        Note that it may accept additional parameters as kwargs.
        """
        class_name = to_pascal(self.dataset_generated__background_function)

        distribution_class = _retrieve_from_module(background, class_name)
        
        return distribution_class(self._dataset__number_of_dimensions, **self.dataset_generated__background_parameters)

    @property
    def dataset_generated__background_pdf(self) -> Callable[[FLOAT_OR_ARRAY], FLOAT_OR_ARRAY]:
        return lambda x: self.__dataset_generated__background_distribution.pdf(
            x / np.exp(self.dataset__induced_shape_nuisance_value),
        )

    @property
    def __dataset_generated__signal_distribution(self) -> DataDistribution:
        """
        Get the signal PDF function based on the configuration.
        Note that it may accept additional parameters as kwargs.
        """
        class_name = to_pascal(self.dataset__signal_data_generation_function)

        distribution_class = _retrieve_from_module(signal, class_name, signal.NoSignal)

        return distribution_class(self._dataset__number_of_dimensions, **self.dataset__signal_parameters)

    @property
    def dataset_generated__signal_pdf(self) -> Callable[[FLOAT_OR_ARRAY], FLOAT_OR_ARRAY]:
        return lambda x: self.__dataset_generated__signal_distribution.pdf(
            x / np.exp(self.dataset__induced_shape_nuisance_value),
        )

    @property
    def dataset__data(self) -> DataSet:
        background = self.__dataset_generated__background_distribution.generate_amount(
            amount=self.dataset__number_of_background_events,
        )
        signal = self.__dataset_generated__signal_distribution.generate_amount(
            amount=self.dataset__number_of_signal_events,
        )
        return background + signal
    
    def __post_init__(self):
        super().__post_init__()

        # dataset_generated__background_function has to be defined by config,
        # but class hierarchy forces to set a default. Hence, we check it here
        assert self.dataset_generated__background_function, \
            "dataset_generated__background_function must be defined in the configuration"

@dataclass
class DatasetConfig:
    
    dataset__number_of_dimensions: int
    dataset__definitions: List[Dict[str, Any]]
    
    # Properties to avoid being documented in context
    @property
    def _dataset__types(self) -> Dict[str, Type[DatasetParameters]]:
        return {cls.DATASET_PARAMTER_TYPE_NAME(): cls for cls in DatasetParameters.__subclasses__()}
    @property
    def _dataset__name_property(self) -> str:
        return "name"
    @property
    def _dataset__type_property(self) -> str:
        return "type"
    @property
    def _dataset__names(self) -> List[str]:
        return [user_dataset_definitions[self._dataset__name_property] for user_dataset_definitions in self.dataset__definitions]

    def _dataset__parameters(self, name: str) -> DatasetParameters:
        # Create datasets definitions from the input arguments
        for user_dataset_definitions in self.dataset__definitions:
            try:
                dataset_name = user_dataset_definitions[self._dataset__name_property]
                dataset_type = user_dataset_definitions[self._dataset__type_property]
            except KeyError:
                raise KeyError(f"Dataset definition must contain '{self._dataset__name_property}' and '{self._dataset__type_property}' keys")
        
            if dataset_name == name:
                try:
                    dataset_class = self._dataset__types[dataset_type]
                except KeyError:
                    raise KeyError(f"Dataset type '{dataset_type}' not defined")

                set = dataset_class(**user_dataset_definitions)
                set._dataset__number_of_dimensions = self.dataset__number_of_dimensions
                return set

        raise KeyError(f"Dataset '{name}' not defined")

    def get_parameters(self, item: str) -> DatasetParameters:
        try:
            return self._dataset__parameters(item)
        except KeyError:
            raise KeyError(f"Dataset '{item}' not defined")
