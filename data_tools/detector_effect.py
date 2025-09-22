from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt

from data_tools.data_utils import DataSet, create_bins
from data_tools.dataset_config import DatasetParameters
from data_tools.detector import error
from data_tools.detector.detector_config import DetectorConfig
from data_tools.detector.efficiency import shapes, uncertainty
from data_tools.detector.efficiency.shapes import DETECTOR_EFFICIENCY_TYPE
from data_tools.detector.efficiency.uncertainty import DETECTOR_EFFICIENCY_UNCERTAINTY_TYPE
from data_tools.detector.error import DETECTOR_ERROR_TYPE
from frame.context.execution_context import ExecutionContext
from frame.module_retriever import retrieve_from_module
from plot.plots import plot_data_generation_sliced


class DetectorEffect:
    """
    Responsible for the interaction between the data and the detector.
    Exported functions are divided into 2 parts:
    - The application of the detector effects on the dataset, done with perfect knowledge using "_true_efficiency"
    - The attempt to correct for the detector effects, done over binned data and with deviations from
     the true efficiency due to simulated uncertainty. This is done using "_binned_efficiency_uncertainty"
    Use them in the proper part of the generation/prediction process.
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

        # Detector dimensions and binning
        assert (ndim := len(self._config.detector__detect_observable_names)) == len(self._config.detector__binning_maxima), "Detector binning dimensions don't match"
        assert ndim == len(self._config.detector__binning_maxima), "Detector binning dimensions don't match"
        assert ndim == len(self._config.detector__binning_number_of_bins), "Detector binning dimensions don't match"
        self._ndim = ndim
        self._observable_names = self._config.detector__detect_observable_names
        self._numbers_of_bins = self._config.detector__binning_number_of_bins

        self._dimensional_bin_centers = {}
        self._dimensional_bin_edges = {}
        for i, obs in enumerate(self._observable_names):
            bin_edges, bin_centers = \
                create_bins(
                    xmin=self._config.detector__binning_minima[i],
                    xmax=self._config.detector__binning_maxima[i],
                    nbins=self._config.detector__binning_number_of_bins[i],
                )
            self._dimensional_bin_centers[obs] = bin_centers
            self._dimensional_bin_edges[obs] = bin_edges

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
        # Detector effects on the data
        self._true_efficiency = self.__retrieve_detector_efficiency_filter(dataset_parameters.dataset__detector_efficiency)
        self._error = self.__get_detector_error_inducer(dataset_parameters.dataset__detector_error)

        # Statistics reconstruction mechanism
        self._efficiency_uncertainty = self.__retrieve_detector_efficiency_uncertainty_modifier(
            dataset_parameters.dataset__detector_efficiency_uncertainty
        )

        # finally, finish updating internal state
        self.__dataset_parameters_for_detection = dataset_parameters

    @property
    def _uncertain_efficiency(self) -> DETECTOR_EFFICIENCY_TYPE:
        return self._efficiency_uncertainty(self._true_efficiency)

    ## Data correction - uses theoretical knowledge only
    @property
    def _binned_uncertain_efficiency_compensator(self) -> Callable[[DataSet], np.ndarray]:

        def __compensator(x: DataSet) -> np.ndarray:
            bin_centers = self.get_event_bin_centers(x)
            return np.ones(shape=(x.n_samples,)) / self._uncertain_efficiency(bin_centers)

        return __compensator

    # Exported functions - uses DataSet
    def generate_true_efficiency_filter(self, dataset: DataSet) -> np.ndarray:
        """
        Generate a filter for the dataset based on the true efficiency.
        """
        dataset_efficiency = self._true_efficiency(dataset._data)
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
        for obs in self._observable_names:
            max_bin_index = len(self._dimensional_bin_centers[obs]) - 1  # last bin is open-ended
            dim_bin_indices = np.clip(np.expand_dims(np.digitize(
                events.slice_along_observable_names(obs),
                self._dimensional_bin_edges[obs],
            ), axis=1), a_min=0, a_max=max_bin_index)
            bin_center_indices.append(dim_bin_indices)
            bin_centered_events.append(np.array(
                self._dimensional_bin_centers[obs][dim_bin_indices]
            ))

        if indexed:
            return np.column_stack(bin_center_indices)
        else:
            return np.column_stack(bin_centered_events)

    def affect_and_compensate(
            self,
            dataset: DataSet,
            dataset_parameters: DatasetParameters,
            is_display: bool = False,
        ) -> DataSet:
        # For graphing purposes only
        if is_display:
            original_dataset = dataset.create_copy()

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

        # Leave compensating weights that are approximate opposite to the efficiency
        compensating_weights = self._binned_uncertain_efficiency_compensator(affected_dataset)
        affected_dataset._weight_mask *= compensating_weights

        if is_display:
            figure = plot_data_generation_sliced(
                context=self._context,
                original_sample=original_dataset,
                processed_sample=affected_dataset,
                bins=self._dimensional_bin_centers[self._context.config.detector__detect_observable_names[0]],
                xlabel=f"{self._context.config.detector__detect_observable_names[0]}",
            )
            self._context.save_and_document_figure(
                figure, self._context.unique_out_dir / f"{dataset_parameters.name}_data_process_plot.png"
            )

        return affected_dataset