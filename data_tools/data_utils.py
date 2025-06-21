from __future__ import annotations

from typing import Callable, List, Optional, Tuple, Union

from data_tools.detector import error
from data_tools.detector.efficiency import shapes, uncertainty
from data_tools.detector.efficiency.shapes import DETECTOR_EFFICIENCY_TYPE
from data_tools.detector.efficiency.uncertainty import DETECTOR_EFFICIENCY_UNCERTAINTY_TYPE
from data_tools.detector.error import DETECTOR_ERROR_TYPE
from frame.module_retriever import retrieve_from_module
import numpy as np
from numpy.random import default_rng
import numpy.typing as npt


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
            efficiency_function: str,
            binning_minima: List[int],
            binning_maxima: List[int],
            number_of_bins: List[int],
            efficiency_uncertainty_function: str,
            error_function: str,
        ):
        # Detector effects on the data
        self._true_efficiency = self.__retrieve_detector_efficiency_filter(efficiency_function)
        self._error = self.__get_detector_error_inducer(error_function)

        # Detector dimensions and binning
        assert (ndim := len(binning_minima)) == len(binning_minima), "Detector binning dimensions don't match"
        assert ndim == len(number_of_bins), "Detector binning dimensions don't match"
        self._ndim = ndim

        self._dimensional_bin_centers = []
        self._dimensional_bin_edges = []
        for i in range(ndim):
            bin_edges, bin_centers = \
                create_bins(xmin=binning_minima[i], xmax=binning_maxima[i], nbins=number_of_bins[i])
            self._dimensional_bin_centers.append(bin_centers)
            self._dimensional_bin_edges.append(bin_edges)
        
        # Statistics reconstruction mechanism
        self._efficiency_uncertainty = self.__retrieve_detector_efficiency_uncertainty_modifier(efficiency_uncertainty_function)

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
    ) -> npt.NDArray:
        
        bin_centered_events = []
        for d in range(self._ndim):
            max_bin_index = len(self._dimensional_bin_centers[d]) - 1  # last bin is open-ended
            dim_bin_indices = np.clip(np.expand_dims(np.digitize(
                events.slice_along_observable_indices(d),
                self._dimensional_bin_edges[d],
            ), axis=1), a_min=0, a_max=max_bin_index)
            bin_centered_events.append(np.array(
                self._dimensional_bin_centers[d][dim_bin_indices]
            ))

        return np.column_stack(bin_centered_events)

    def affect_and_compensate(self, dataset: DataSet) -> DataSet:
        filter = self.generate_true_efficiency_filter(dataset)
        affected_dataset = dataset.filter(filter)

        errors = self.generate_errors(affected_dataset)
        affected_dataset._data += errors

        compensating_weights = self._binned_uncertain_efficiency_compensator(affected_dataset)
        affected_dataset._weight_mask *= compensating_weights

        return affected_dataset


class DataSet:
    """
    A class representing a dataset of events.

    Each row in the stored _data is a single event. The whole 2D table represents the
    collection of them.
    """
    
    def __init__(self, data: npt.NDArray, observable_names: Optional[List[str]] = None):
        """
        Data has to be a 2D array
        """
        if data.ndim == 1:
            self._data = np.expand_dims(data, axis=1)
        elif data.ndim == 2:
            self._data = data
        else:
            raise ValueError(f"Data must be a 0D, 1D, or 2D array, but got {data.ndim} dimensions.")
        self._weight_mask = np.ones((self.n_samples,))
        self._observable_names = observable_names if observable_names is not None else [f"param_{i}" for i in range(self.n_observables)]

    def __add__(self, other: DataSet) -> DataSet:
        if self.empty:
            return other
        if other.empty:
            return self
        if self._observable_names != other._observable_names:
            raise ValueError("Observable names do not match between datasets.")
        
        _data = np.concatenate((self._data, other._data), axis=0)
        _weight_mask = np.concatenate((self._weight_mask, other._weight_mask), axis=0)

        result = DataSet(_data, observable_names=self._observable_names)
        result._weight_mask = _weight_mask
        return result

    def __getitem__(self, item: Union[int, slice, npt.NDArray]) -> DataSet:
        result = DataSet(self._data[item, :], observable_names=self._observable_names)
        result._weight_mask = self._weight_mask[item]
        return result

    @property
    def n_observables(self) -> int:
        if self._data.size == 0:
            return 0
        return self._data.shape[1]

    @property
    def n_samples(self):
        return self._data.shape[0]
    
    @property
    def corrected_n_samples(self) -> float:
        return float(np.sum(self._weight_mask))
    
    @property
    def empty(self) -> bool:
        return self.n_samples == 0

    @property
    def histogram_weight_mask(self) -> np.ndarray:
        return np.expand_dims(self._weight_mask, axis=1)

    def slice_along_observable_indices(self, indices: Union[int, slice, npt.NDArray]) -> np.ndarray:
        """
        Get a slice of all events along a single dimension.
        """
        return self._data[:, indices]
    
    def slice_along_observable_names(self, observables: List[str]) -> npt.NDArray:
        indices = [self._observable_names.index(obs) for obs in observables]
        return self.slice_along_observable_indices(np.array(indices))
    
    def filter(self, filter: np.ndarray) -> DataSet:
        """
        Filter the dataset according to a boolean mask.
        """
        filtered_data = self._data[filter]
        filtered_weight_mask = self._weight_mask[filter]

        result = DataSet(filtered_data, observable_names=self._observable_names)
        result._weight_mask = filtered_weight_mask
        return result


def resample(
        source_dataset: DataSet,
        n_samples: int,
        replacement: bool = True
    ) -> Tuple[DataSet, DataSet]:
    """
    Chooses a dataset randomly from the source distribution.
    
    Returns: the sampled dataset and the remaining data, by resampling
    specification.
    
    If no replacement, the number of samples can't be larger than the
    source distribution itself.
    """

    rng = default_rng()
    idx = rng.choice(
        source_dataset.n_samples,
        size=n_samples,
        replace=replacement,
    )

    sample = source_dataset[idx]
    if replacement:
        remainder = source_dataset
    else:
        rest_idx = np.array(list(set(range(source_dataset.n_samples)) - set(idx)), dtype=int)
        remainder = source_dataset[rest_idx]

    return sample, remainder


def create_slice_containing_bins(
        datasets: List[DataSet],
        nbins = 30,
        along_dimension: int = 0,
) -> Tuple[npt.NDArray, npt.NDArray]:

    # limits    
    xmin = 0
    xmax = np.max([np.max(dataset.slice_along_observable_indices(along_dimension)) for dataset in datasets])

    return create_bins(
        xmin=xmin,
        xmax=xmax,
        nbins=nbins,
    )

def create_bins(
        xmin: float,
        xmax: float,
        nbins: int,
) -> Tuple[npt.NDArray, npt.NDArray]:
    
    bins = np.linspace(xmin, xmax, nbins + 1)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    return bins, bin_centers


def create_slice_containing_bins(
        datasets: List[DataSet],
        nbins = 100,
        along_dimension: int = 0,
):
    # limits    
    xmin = 0
    xmax = np.max([np.max(dataset.slice_along_observable_indices(along_dimension)) for dataset in datasets])

    return create_bins(xmin=xmin, xmax=xmax, nbins=nbins)
