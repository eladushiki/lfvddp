from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union
from urllib.parse import urlparse

import awkward as ak
from camel_converter import to_pascal

from data_tools.CMS_open_data import parse_CMS_open_data_sources_json
from data_tools.data_utils import DataSet
from data_tools.event_generation import background, signal
from data_tools.event_generation.distribution import DataDistribution
from data_tools.event_generation.types import FLOAT_OR_ARRAY
from frame.file_system.numpy_events import load_numpy_events
from frame.file_system.root_reader import load_root_events
from frame.file_system.textual_data import load_dict_from_json, read_text_file_lines
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
    dataset__number_of_background_events: int = field(default=None)  # in the case of loaded datasets, None loads the full amount


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
        if not isfile(self.dataset_loaded__file_name):
            try:
                urlparse(self.dataset_loaded__file_name)
            except ValueError:
                raise FileNotFoundError(f"Loaded file '{self.dataset_loaded__file_name}' does not exist, nor it's a valid URL.")

    # Data source
    dataset_loaded__file_name: str = field(default="")
    dataset_loaded__file_parameters: Dict[str, Any] = field(default_factory=dict)

    # Resampling settings
    dataset_loaded__resample_is_resample: bool = field(default=False)
    dataset_loaded__resample_is_replacement: bool = field(default=False)

    @property
    def dataset__data(self) -> DataSet:
        """
        Load the data from the specified file, and update the internal
        state of loaded data to match resampling settings.
        """
        loaded_dataset = self.__load_dataset(
            self.dataset_loaded__file_name,
            self.dataset__number_of_background_events
        )

        if loaded_dataset.dim != self._dataset__number_of_dimensions:
            raise ValueError(f"Loaded dataset dimensions {loaded_dataset.dim} do not match expected dimensions {self._dataset__number_of_dimensions}.")

        if loaded_dataset.n_samples < self.dataset__number_of_background_events:
            raise ValueError(f"Loaded dataset has only {loaded_dataset.n_samples} samples, "\
                f"but requested {self.dataset__number_of_background_events} samples.")
                
        return loaded_dataset
      
    def __load_dataset(self, path: str, number_of_events: Optional[int] = None) -> DataSet:
        """
        Load data from the specified file.
        If file is a text or json file, assume it contains a list
        of ROOT files, then load them recursively (not supposed
        to contain more than one hierarchy).
        """
        file_extension = Path(path).suffix

        if file_extension == ".npy":
            loaded_dataset = load_numpy_events(path, number_of_events)
        
        elif file_extension == ".root":
            loaded_dataset = load_root_events(
                XRootD_url=path,
                stop=number_of_events,
                **self.dataset_loaded__file_parameters,
            )
        
        else:  # Assuming the file contains a list of root files to load
            if file_extension == ".json":
                json_params = load_dict_from_json(Path(path))
                source_uri_list = parse_CMS_open_data_sources_json(json_params)
            
            elif file_extension == ".txt":
                source_uri_list = read_text_file_lines(Path(path))

            else:
                raise ValueError(f"Unsupported file format: {file_extension} for data source file, got {file_extension}.")

            if not source_uri_list:
                raise ValueError(f"No source URIs found in the file: {path}. Please check the file content.")
            
            loaded_datasets = []
            for source_uri in source_uri_list:
                additional_events = self.__load_dataset(source_uri, number_of_events)
                
                if  number_of_events is not None:
                    number_of_events -= additional_events.n_samples
                    if number_of_events <= 0:
                        break

            loaded_dataset = sum(loaded_datasets[1:], loaded_datasets[0])
  
        return loaded_dataset


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
