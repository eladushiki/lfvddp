from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Tuple, Union

import pandas as pd

import numpy as np
from numpy.random import default_rng
import numpy.typing as npt


class DataSet:
    """
    A class representing a dataset of events.

    Each row in the stored _data is a single event. The whole 2D table represents the
    collection of them.
    """
    
    def __init__(self, data: Optional[Union[npt.NDArray, pd.DataFrame]] = None, observable_names: Optional[List[str]] = None):
        if isinstance(data, np.ndarray):
            if data is None or len(data) == 0:
                self._data = pd.DataFrame()
            elif data.ndim == 1 or data.ndim == 2:
                self._data = pd.DataFrame(data)
            else:
                raise ValueError(f"Data as numpy array must be a 0D, 1D, or 2D array, but got {data.ndim} dimensions.")
        elif isinstance(data, pd.DataFrame):
            self._data = data
        else:
            raise TypeError(f"Unacceptable typing for data, {type(data)}")

        if observable_names is not None:
            self._data.columns = observable_names
        else:
            self._data.columns = [f"param_{i}" for i in range(self.n_observables)]
        
        self._weight_mask = np.ones((self.n_samples,))

    def __add__(self, other: DataSet) -> DataSet:
        if self.empty:
            return other
        if other.empty:
            return self
        if self.observable_names != other.observable_names:
            raise ValueError("Observable names do not match between datasets.")
        
        _data = pd.concat((self._data, other._data), axis=0)
        _data.reset_index(level=0, drop=True, inplace=True)
        _weight_mask = np.concatenate((self._weight_mask, other._weight_mask), axis=0)

        result = DataSet(_data, observable_names=self.observable_names)
        result._weight_mask = _weight_mask
        return result

    def __getitem__(self, item: Union[int, slice, npt.NDArray]) -> DataSet:
        result = DataSet(
            pd.DataFrame(self._data.iloc[item, :]),
            observable_names=self.observable_names,
        )
        result._weight_mask = self._weight_mask[item]
        return result

    def create_copy(self) -> DataSet:
        copy = DataSet(self._data.copy(), observable_names=self.observable_names)
        copy._weight_mask = self._weight_mask.copy()
        return copy

    @property
    def observable_names(self) -> List[str]:
        return self._data.columns.tolist()
    
    @property
    def n_observables(self) -> int:
        return len(self.observable_names)

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
    def events(self) -> npt.NDArray:
        return self._data.to_numpy()

    @property
    def histogram_weight_mask(self) -> np.ndarray:
        return np.expand_dims(self._weight_mask, axis=1)

    def slice_along_observable_indices(self, indices: Optional[Union[int, slice, npt.NDArray]] = None) -> npt.NDArray:
        """
        Get a slice of all events along a single dimension.
        """
        if indices is None:
            indices = 0

        return self.slice_along_observable_names(self.observable_names[indices])

    def slice_along_observable_names(self, observables: Union[str, List[str]]) -> npt.NDArray:
        try:
            return self._data[observables].to_numpy()
        except KeyError as e:
            raise KeyError(f"One or more observable names not found in dataset: {observables}") from e
    
    def filter(self, filter: np.ndarray) -> DataSet:
        """
        Filter the dataset according to a boolean mask.
        """
        filtered_data = self._data.iloc[filter, :]
        filtered_weight_mask = self._weight_mask[filter]

        result = DataSet(filtered_data, observable_names=self.observable_names)
        result._weight_mask = filtered_weight_mask
        return result

    def filter_observable_names(self, observables: Union[str, List[str]]) -> DataSet:
        filtered_dataset = DataSet(
            self.slice_along_observable_names(observables),
            observable_names=[observables] if isinstance(observables, str) else observables
        )
        filtered_dataset._weight_mask = deepcopy(self._weight_mask)
        return filtered_dataset


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
