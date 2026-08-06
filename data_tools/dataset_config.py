import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from inspect import isabstract
from os.path import isfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
from urllib.parse import urlparse

import numpy as np

from data_tools.CMS_open_data import parse_CMS_open_data_sources_json
from data_tools.data_utils import DataSet
from data_tools.event_generation import background, signal
from data_tools.event_generation.distribution import (
    DataDistribution,
    GeneratorSelectionConfig,
    describe_generator_selection,
    generator_selection_as_config,
    normalize_generator_selection,
    resolve_generator,
    validate_generated_dataset,
)
from data_tools.event_generation.types import FLOAT_OR_ARRAY
from frame.file_system.numpy_events import load_numpy_events
from frame.file_system.root_reader import load_root_events
from frame.file_system.textual_data import load_dict_from_json, read_text_file_lines


@dataclass
class DatasetParameters(ABC):
    
    # For documentation purposes
    name: str
    type: str
    category: DataSet.DataSetCategory

    # Background parameters
    dataset__mean_number_of_background_events: int = field(default=None)

    # Detector simulation
    dataset__detector_efficiency: str = field(default="")
    dataset__detector_efficiency_uncertainty: str = field(default="")
    dataset__detector_error: str = field(default="")

    # Induced nuisance parameters (relevant for old NPLM implementation)
    dataset__induced_shape_nuisance_value: float = field(default=0.0)
    dataset__induced_norm_nuisance_value: float = field(default=0.0)

    # Created automatically
    ## Picked poissonically based on mean numbers
    dataset__number_of_background_events: int = field(default=None)  # in the case of loaded datasets, None loads the full amount

    @classmethod
    @abstractmethod
    def DATASET_PARAMETER_TYPE_NAME(cls) -> str:
        pass

    @property
    @abstractmethod
    def dataset__data(self) -> Tuple[DataSet, DataSet]:
        pass

    @property
    @abstractmethod
    def dataset__background_source_type(self) -> str:
        pass

    @property
    @abstractmethod
    def dataset__background_source(self) -> str:
        pass

    @property
    @abstractmethod
    def dataset__has_signal(self) -> bool:
        pass

    @property
    @abstractmethod
    def dataset__signal_source(self) -> str:
        pass

    @property
    @abstractmethod
    def dataset__signal_description(self) -> str:
        pass

    def __post_init__(self):
        # Poisson distribution of event numbers per run given mean
        if not self.dataset__number_of_background_events:
            assert self.dataset__mean_number_of_background_events is not None, \
                "Number of background events must be defined in the configuration, either directly or via mean."
            self.dataset__number_of_background_events = np.random.poisson(
                lam=self.dataset__mean_number_of_background_events * np.exp(self.dataset__induced_norm_nuisance_value),
                size=1,
            ).item() if self.dataset__mean_number_of_background_events > 0 else 0

        if isinstance(self.category, str):
            self.category = DataSet.DataSetCategory.from_string(self.category)

    @property
    @abstractmethod
    def dataset__number_of_dimensions(self) -> int:
        """Number of observables / dimensions in the dataset."""
        pass



@dataclass
class DatasetWithGeneratedSignalParameters(DatasetParameters, ABC):
    """Dataset source parameters with an optional generated signal overlay."""

    dataset__signal_generator: Optional[GeneratorSelectionConfig] = field(default=None)
    dataset__signal_number_of_events_to_generate: int = field(default=None)
    dataset__mean_number_of_signal_events: int = field(default=0)
    dataset__number_of_signal_events: int = field(default=None)

    _dataset__signal_source: str = field(init=False, default="", repr=False)
    _dataset__signal_description: str = field(init=False, default="", repr=False)

    def __post_init__(self):
        super().__post_init__()

        if not self.dataset__number_of_signal_events:
            self.dataset__number_of_signal_events = np.random.poisson(
                lam=self.dataset__mean_number_of_signal_events
                * np.exp(self.dataset__induced_norm_nuisance_value),
                size=1,
            ).item() if self.dataset__mean_number_of_signal_events > 0 else 0

        if self.dataset__signal_number_of_events_to_generate:
            if (
                self.dataset__signal_number_of_events_to_generate
                < self.dataset__number_of_signal_events
            ):
                raise ValueError(
                    "Not sufficient number of generated signal events for the "
                    f"requested {self.dataset__number_of_signal_events} events."
                )
        else:
            self.dataset__signal_number_of_events_to_generate = (
                self.dataset__number_of_signal_events
            )

        if self.dataset__signal_generator is None:
            if self.dataset__signal_number_of_events_to_generate > 0:
                raise ValueError(
                    "dataset__signal_generator must be configured when signal "
                    "events are requested."
                )
            return

        self.dataset__signal_generator = normalize_generator_selection(
            self.dataset__signal_generator
        )
        self._dataset__signal_source = json.dumps(
            generator_selection_as_config(self.dataset__signal_generator),
            sort_keys=True,
        )
        self._dataset__signal_description = describe_generator_selection(
            self.dataset__signal_generator
        )

    @property
    def dataset__has_signal(self) -> bool:
        return bool(
            self.dataset__number_of_signal_events
            or self.dataset__mean_number_of_signal_events
        )

    @property
    def dataset__signal_source(self) -> str:
        return self._dataset__signal_source

    @property
    def dataset__signal_description(self) -> str:
        return self._dataset__signal_description

    @property
    def _dataset__signal_distribution(self) -> DataDistribution:
        if self.dataset__signal_generator is None:
            return signal.NoSignal(self.dataset__number_of_dimensions)
        return resolve_generator(
            signal,
            self.dataset__signal_generator,
            self.dataset__number_of_dimensions,
        )

    @property
    def dataset_generated__signal_pdf(self) -> Callable[[FLOAT_OR_ARRAY], FLOAT_OR_ARRAY]:
        return lambda x: self._dataset__signal_distribution.pdf(
            x / np.exp(self.dataset__induced_shape_nuisance_value),
        )

    def _dataset__generate_signal(self) -> DataSet:
        generated_signal = self._dataset__signal_distribution.generate_amount(
            amount=self.dataset__signal_number_of_events_to_generate,
        )
        validate_generated_dataset(
            generated_signal,
            self.dataset__signal_number_of_events_to_generate,
            self.dataset__number_of_dimensions,
            "signal",
        )
        return generated_signal

    
@dataclass
class LoadedDatasetParameters(DatasetWithGeneratedSignalParameters):
    
    @classmethod
    def DATASET_PARAMETER_TYPE_NAME(cls) -> str:
        return "loaded"

    @property
    def dataset__background_source_type(self) -> str:
        return self.DATASET_PARAMETER_TYPE_NAME()

    @property
    def dataset__background_source(self) -> str:
        return self.dataset_loaded__file_name

    def __post_init__(self):
        super().__post_init__()

        # Make sure the file exists
        if not isfile(self.dataset_loaded__file_name):
            try:
                urlparse(self.dataset_loaded__file_name)
            except ValueError:
                raise FileNotFoundError(f"Loaded file '{self.dataset_loaded__file_name}' does not exist, nor it's a valid URL.")

        assert self.dataset_loaded__observable_naming is not None, \
            "dataset_loaded__observable_naming must be defined in the configuration"

    # Data source
    dataset_loaded__file_name: str = field(default=None)
    dataset_loaded__event_amount_load_limit: Optional[int] = field(default=None)

    # Names of observables to load should be defined by the convension {name_in_dataset: name_in_program}
    dataset_loaded__observable_naming: Dict[str, str] = field(default_factory=dict)

    # Tampering mechanics
    dataset_loaded__cut: Optional[str] = field(default=None)
    dataset_loaded__aliases: Optional[Dict[str, str]] = field(default=None)

    # Resampling settings
    dataset_loaded__resample_is_resample: bool = field(default=False)
    dataset_loaded__resample_is_replacement: bool = field(default=False)

    @property
    def dataset_loaded__observable_to_load(self) -> Iterable[str]:
        return self.dataset_loaded__observable_naming.keys()

    @property
    def dataset_loaded__observable_names(self) -> Iterable[str]:
        return self.dataset_loaded__observable_naming.values()

    @property
    def dataset__number_of_dimensions(self) -> int:
        return len(self.dataset_loaded__observable_naming)

    @property
    def dataset__data(self) -> Tuple[DataSet, DataSet]:
        """
        Load the data from the specified file, and update the internal
        state of loaded data to match resampling settings.
        """
        background = self.__load_dataset(
            self.dataset_loaded__file_name,
            self.dataset_loaded__event_amount_load_limit
        )
        signal = self._dataset__generate_signal()
        if not signal.empty:
            signal.observable_names = background.observable_names
        return background, signal
        
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
                branch_names=self.dataset_loaded__observable_to_load,
                observable_renames=self.dataset_loaded__observable_naming,
                cut=self.dataset_loaded__cut,
                aliases=self.dataset_loaded__aliases,
                stop=number_of_events,
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
class GeneratedDatasetParameters(DatasetWithGeneratedSignalParameters):

    @classmethod
    def DATASET_PARAMETER_TYPE_NAME(cls) -> str:
        return "generated"

    _dataset__background_source: str = field(init=False, default="", repr=False)

    @property
    def dataset__background_source_type(self) -> str:
        return self.DATASET_PARAMETER_TYPE_NAME()

    @property
    def dataset__background_source(self) -> str:
        return self._dataset__background_source
    
    dataset_generated__number_of_dimensions: int = field(default=1)

    @property
    def dataset__number_of_dimensions(self) -> int:
        return self.dataset_generated__number_of_dimensions

    @dataset__number_of_dimensions.setter
    def dataset__number_of_dimensions(self, value: int) -> None:
        self.dataset_generated__number_of_dimensions = value
    
    # Additional background parameters
    # This is the defining attribute for the subclass
    dataset_generated__background_generator: Optional[GeneratorSelectionConfig] = field(
        default=None
    )

    @property
    def __dataset_generated__background_distribution(self) -> DataDistribution:
        return resolve_generator(
            background,
            self.dataset_generated__background_generator,
            self.dataset__number_of_dimensions,
        )

    @property
    def dataset_generated__background_pdf(self) -> Callable[[FLOAT_OR_ARRAY], FLOAT_OR_ARRAY]:
        return lambda x: self.__dataset_generated__background_distribution.pdf(
            x / np.exp(self.dataset__induced_shape_nuisance_value),
        )

    @property
    def dataset__data(self) -> Tuple[DataSet, DataSet]:
        background = self.__dataset_generated__background_distribution.generate_amount(
            amount=self.dataset__number_of_background_events,
        )
        validate_generated_dataset(
            background,
            self.dataset__number_of_background_events,
            self.dataset__number_of_dimensions,
            "background",
        )
        signal = self._dataset__generate_signal()
        return background, signal
    
    def __post_init__(self):
        super().__post_init__()

        if self.dataset_generated__background_generator is None:
            raise ValueError(
                "dataset_generated__background_generator must be configured for "
                "generated datasets."
            )
        self.dataset_generated__background_generator = normalize_generator_selection(
            self.dataset_generated__background_generator
        )
        self._dataset__background_source = json.dumps(
            generator_selection_as_config(
                self.dataset_generated__background_generator
            ),
            sort_keys=True,
        )

        if self.dataset__number_of_dimensions <= 0:
            raise ValueError(
                "dataset__number_of_dimensions must be defined and greater than 0"
            )

@dataclass
class DatasetConfig:
    
    dataset__definitions: List[Dict[str, Any]]
    _dataset__parameters_by_category: Dict[
        DataSet.DataSetCategory, DatasetParameters
    ] = field(init=False, default_factory=dict, repr=False)

    # Properties to avoid being documented in context
    @property
    def _dataset__types(self) -> Dict[str, Type[DatasetParameters]]:
        parameter_types = {}
        subclasses = list(DatasetParameters.__subclasses__())
        while subclasses:
            subclass = subclasses.pop()
            subclasses.extend(subclass.__subclasses__())
            if isabstract(subclass):
                continue
            parameter_types[subclass.DATASET_PARAMETER_TYPE_NAME()] = subclass
        return parameter_types

    @property
    def _dataset__name_property(self) -> str:
        return "name"

    @property
    def _dataset__type_property(self) -> str:
        return "type"

    @property
    def _dataset__category_property(self) -> str:
        return "category"

    @property
    def dataset_parameters(self) -> List[DatasetParameters]:
        self._ensure_dataset_parameters_loaded()
        return list(self._dataset__parameters_by_category.values())

    def load_dataset_parameters(self) -> None:
        """Construct each configured dataset once after the context RNG is seeded."""
        dataset_types = self._dataset__types
        parameters_by_category: Dict[
            DataSet.DataSetCategory, DatasetParameters
        ] = {}
        for user_dataset_definitions in self.dataset__definitions:
            try:
                dataset_type = user_dataset_definitions[self._dataset__type_property]
                dataset_category = user_dataset_definitions[self._dataset__category_property]
                user_dataset_definitions[self._dataset__name_property]
            except KeyError as error:
                raise KeyError(
                    "Dataset definition must contain 'name', 'type', and "
                    "'category' keys."
                ) from error

            try:
                dataset_class = dataset_types[dataset_type]
            except KeyError as error:
                raise KeyError(f"Dataset type '{dataset_type}' not defined") from error

            category = DataSet.DataSetCategory.from_string(dataset_category)
            if category == DataSet.DataSetCategory.UNDEFINED:
                raise ValueError(f"Dataset category '{dataset_category}' is not defined")
            if category in parameters_by_category:
                raise ValueError(f"Duplicate dataset category '{dataset_category}'")

            parameters_by_category[category] = dataset_class(
                **user_dataset_definitions,
            )

        self._dataset__parameters_by_category = parameters_by_category

    def _ensure_dataset_parameters_loaded(self) -> None:
        if not self._dataset__parameters_by_category:
            self.load_dataset_parameters()

    def get_parameters(self, item: DataSet.DataSetCategory) -> DatasetParameters:
        self._ensure_dataset_parameters_loaded()
        try:
            return self._dataset__parameters_by_category[item]
        except KeyError as error:
            raise KeyError(f"Dataset '{item}' not defined") from error
