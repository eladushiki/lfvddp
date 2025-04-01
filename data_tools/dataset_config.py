from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Type

import numpy as np

@dataclass
class DatasetParameters(ABC):
    
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
    dataset__background_data_generation_function: str
    dataset__number_of_background_events: int

    # Signal parameters
    dataset__number_of_signal_events: int
    dataset__signal_location: int
    
    # Detector simulation
    dataset__detector_efficiency: str
    dataset__detector_error: str

    # Induced nuisance parameters
    dataset__induced_nuisances_shape_sigma: float             # shape nuisance sigma  # todo: convert to a list to enable any number of those
    dataset__induced_nuisances_shape_mean_sigmas: float       # shape nuisance reference, in terms of std
    dataset__induced_nuisances_shape_reference_sigmas: float  # norm nuisance reference, in terms of std
    
    dataset__induced_nuisances_norm_sigma: float              # norm nuisance sigma
    dataset__induced_nuisances_norm_mean_sigmas: float        # in terms of std
    dataset__induced_nuisances_norm_reference_sigmas: float   # in terms of std

    # Resampling settings
    dataset__resample_is_resample: bool
    dataset__resample_label_method: str
    dataset__resample_method_type: str
    dataset__resample_is_replacement: bool

    dataset__function_specific_additional_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    
    dataset__definitions: List[Dict[str, Any]]

    _dataset__parameters: Dict[str, DatasetParameters] = field(default_factory=dict)
    
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

    def __post_init__(self):
        # Create datasets definitions from the input arguments
        for user_dataset_definitions in self.dataset__definitions:
            try:
                dataset_name = user_dataset_definitions[self._dataset__name_property]
                dataset_type = user_dataset_definitions[self._dataset__type_property]
            except KeyError:
                raise KeyError(f"Dataset definition must contain '{self._dataset__name_property}' and '{self._dataset__type_property}' keys")
        
            try:
                dataset_class = self._dataset__types[dataset_type]
            except KeyError:
                raise KeyError(f"Dataset type '{dataset_type}' not defined")

            del user_dataset_definitions[self._dataset__name_property]
            del user_dataset_definitions[self._dataset__type_property]
            self._dataset__parameters[dataset_name] = dataset_class(**user_dataset_definitions)

        # Avoid duplicate documentation in context, this is included in _dataset__parameters
        del self.dataset__definitions

    def get_parameters(self, item: str) -> DatasetParameters:
        try:
            return self._dataset__parameters[item]
        except KeyError:
            raise KeyError(f"Dataset '{item}' not defined")
