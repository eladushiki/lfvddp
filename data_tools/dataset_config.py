from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import numpy as np

@dataclass
class DatasetParameters(ABC):
    
    _dataset__number_of_dimensions: int = field(init=False)
    
    # For documentation purposes
    name: str
    type: str

    @classmethod
    @abstractmethod
    def DATASET_PARAMTER_TYPE_NAME(cls) -> str:
        pass


@dataclass
class LoadedDatasetParameters(DatasetParameters):
    @classmethod
    def DATASET_PARAMTER_TYPE_NAME(cls) -> str:
        return "loaded"
    
    ## Using real dataset parts implementation are left for the reader.
    # said reader would like to, firsly, implement
    # train__histogram_is_use_analytic: int  # if 1: generate data from pdf

    def __post_init__(self):
        raise NotImplementedError("LoadedDatasetDefinitions is not implemented yet.")


@dataclass
class GeneratedDatasetParameters(DatasetParameters, ABC):

    @classmethod
    def DATASET_PARAMTER_TYPE_NAME(cls) -> str:
        return "generated"
    
    # Background parameters
    # This is the defining attribute for the subclass
    dataset__background_generation_function: str
    dataset__mean_number_of_background_events: int

    # Signal parameters
    dataset__signal_data_generation_function: str
    dataset__mean_number_of_signal_events: int
    dataset__signal_parameters: Dict[str, Any]
    
    # Detector simulation
    dataset__detector_efficiency: str
    dataset__detector_efficiency_uncertainty: str
    dataset__detector_error: str

    # Induced nuisance parameters
    dataset__induced_shape_nuisance_value: float
    dataset__induced_norm_nuisance_value: float

    # Resampling settings
    dataset__resample_is_resample: bool
    dataset__resample_label_method: str
    dataset__resample_method_type: str
    dataset__resample_is_replacement: bool

    dataset__function_specific_additional_parameters: Dict[str, Any] = field(default_factory=dict)
    dataset__number_of_signal_events: int = field(default=None)
    dataset__number_of_background_events: int = field(default=None)

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
