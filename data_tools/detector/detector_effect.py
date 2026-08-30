from typing import Optional, Union

import numpy as np
import numpy.typing as npt

from data_tools.data_generation import DataBatch
from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetParameters
from data_tools.detector import error
from data_tools.detector.detector_config import DetectorConfig
from data_tools.detector.efficiency import shapes, uncertainty
from data_tools.detector.efficiency.shapes import DETECTOR_EFFICIENCY_TYPE
from data_tools.detector.efficiency.uncertainty import (
    DETECTOR_EFFICIENCY_UNCERTAINTY_TYPE,
)
from data_tools.detector.error import DETECTOR_ERROR_TYPE
from frame.context.execution_context import ExecutionContext
from frame.module_retriever import retrieve_from_module


class DetectorEffect:  # TODO: binning functionality should be separated from the detector
    """
    Responsible for the interaction between the data and the detector.
    Applies detector efficiency and measurement error effects to datasets.
    """
    def __init__(
            self,
            context: ExecutionContext,
        ):
        self._context = context
        if not isinstance(self._context.config, DetectorConfig):
            raise TypeError(f"Expected DetectorConfig, got {self._context.config.__class__.__name__}")
        self._config = self._context.config
        self.__dataset_parameters_for_detection = None

        # Detector binning is needed only by the scalar nuisance estimator.
        self._observable_names = self._config.detector__detect_observable_names
        self._numbers_of_bins = self._config.train__nuisance_binning_number_of_bins
        self._dimensional_bin_centers = {}
        self._dimensional_bin_edges = {}
        if not self._config.train__nuisance_is_neural_network:
            for obs in self._observable_names:
                self._dimensional_bin_edges[obs], self._dimensional_bin_centers[obs] = \
                    self._config.observable_bins(obs)

    @retrieve_from_module(shapes, shapes.detector_efficiency_perfect_efficiency)
    def __retrieve_detector_efficiency_filter(self, effect_name: Optional[str]) -> Union[DETECTOR_EFFICIENCY_TYPE, str, None]:
        """
        Detector efficiency indicated the probability for each event (=row) to remain.
        """
        return effect_name

    @retrieve_from_module(uncertainty, uncertainty.detector_uncertainty_no_uncertainty)
    def __retrieve_detector_efficiency_uncertainty_modifier(self, uncertainty: Optional[str]) -> Union[DETECTOR_EFFICIENCY_UNCERTAINTY_TYPE, str, None]:
        """
        Detector efficiency uncertainty.
        """
        return uncertainty

    @retrieve_from_module(error, error.detector_no_error)
    def __get_detector_error_inducer(self, error_name: Optional[str]) -> Union[DETECTOR_ERROR_TYPE, str, None]:
        """
        Detector error returns the same shape as the input.
        """
        return error_name

    @property
    def detection_parameters(self) -> Optional[DatasetParameters]:
        return self.__dataset_parameters_for_detection

    @detection_parameters.setter
    def detection_parameters(self, dataset_parameters: DatasetParameters):
        # Detector effects belong to the detector configuration.  Fall back to
        # legacy dataset fields while old configuration packs are migrated.
        family = getattr(dataset_parameters.category, "name", str(dataset_parameters.category)).split("_")[0].lower()
        if family == "a":
            efficiency = self._config.detector__effect_a_efficiency
            error = self._config.detector__effect_a_error
            uncertainty = self._config.detector__effect_a_efficiency_uncertainty
        elif family == "b":
            efficiency = self._config.detector__effect_b_efficiency
            error = self._config.detector__effect_b_error
            uncertainty = self._config.detector__effect_b_efficiency_uncertainty
        else:
            raise ValueError(f"Unsupported detector dataset family: {family!r}")
        if not any((efficiency, error, uncertainty)):
            efficiency = dataset_parameters.dataset__detector_efficiency
            error = dataset_parameters.dataset__detector_error
            uncertainty = dataset_parameters.dataset__detector_efficiency_uncertainty
        self._true_efficiency = self.__retrieve_detector_efficiency_filter(efficiency)
        self._error = self.__get_detector_error_inducer(error)

        self._efficiency_uncertainty = self.__retrieve_detector_efficiency_uncertainty_modifier(
            uncertainty
        )

        # finally, finish updating internal state
        self.__dataset_parameters_for_detection = dataset_parameters

    @property
    def _uncertain_efficiency(self) -> DETECTOR_EFFICIENCY_TYPE:
        return self._efficiency_uncertainty(self._true_efficiency)

    # Exported functions - uses DataSet
    def get_observable_bins(
        self,
        observable_name: str,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Return the detector bin edges and centers for one observable."""
        try:
            return (
                self._dimensional_bin_edges[observable_name].copy(),
                self._dimensional_bin_centers[observable_name].copy(),
            )
        except KeyError as error:
            raise ValueError(
                f"Observable {observable_name} is not detected by this detector effect."
            ) from error

    def efficiency_values(self, dataset: DataSet) -> np.ndarray:
        """Return the detector efficiency at each dataset point without sampling."""
        if self.detection_parameters is None:
            raise RuntimeError(
                "Detector efficiency cannot be evaluated before detection "
                "parameters are set."
            )
        return np.asarray(self._uncertain_efficiency(dataset._data))

    def generate_true_efficiency_filter(self, dataset: DataSet) -> np.ndarray:
        """
        Generate a filter for the dataset based on the true efficiency.
        """
        dataset_efficiency = self.efficiency_values(dataset)
        return np.random.uniform(size=(dataset.n_samples,)) < dataset_efficiency

    def generate_errors(self, dataset: DataSet) -> np.ndarray:
        """
        Generate errors for the dataset based on the error function.
        """
        return self._error(dataset._data)

    def get_event_bin_centers(
        self,
        events: DataSet,
        indexed: bool = False,
    ) -> npt.NDArray:

        bin_centered_events = []
        bin_center_indices = []
        for obs in events.observable_names:
            max_bin_index = len(self._dimensional_bin_centers[obs]) - 1  # last bin is open-ended
            dim_bin_indices = np.clip(np.expand_dims(np.digitize(
                events.slice_along_observable_names(obs),
                self._dimensional_bin_edges[obs],
            ) - 1, axis=1), a_min=0, a_max=max_bin_index)
            bin_center_indices.append(dim_bin_indices)
            bin_centered_events.append(np.array(
                self._dimensional_bin_centers[obs][dim_bin_indices]
            ))

        if indexed:
            return np.column_stack(bin_center_indices)
        else:
            return np.column_stack(bin_centered_events)

    def affect_batch(
        self,
        batch: DataBatch,
    ) -> DataBatch:
        """
        Apply the detector effect to a batch of datasets.
        """
        affected_datasets = []
        for dataset, dataset_parameters in batch:
            affected_dataset = self.affect_dataset(dataset, dataset_parameters)
            affected_datasets.append((affected_dataset, dataset_parameters))
        return DataBatch(affected_datasets)

    def affect_dataset(
            self,
            dataset: DataSet,
            dataset_parameters: DatasetParameters,
        ) -> DataSet:
        # Update internal state for detection
        self.detection_parameters = dataset_parameters

        # Leave only detected fields
        detected_dataset = dataset.filter_observable_names(self._observable_names)

        # Keep each event by efficiency defined probability
        filter = self.generate_true_efficiency_filter(detected_dataset)
        affected_dataset = detected_dataset.filter(filter)

        # Induce detector errors of the true measurements
        errors = self.generate_errors(affected_dataset)
        affected_dataset._data += errors

        return affected_dataset
