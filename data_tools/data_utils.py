from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd


class DataSet:
    """
    A class representing a dataset of events.

    Each row in the stored _data is a single event. The whole 2D table represents the
    collection of them.
    """
    
    class DataSetCategory(enum.Enum):
        A_SR = 1
        A_CR = 2
        B_SR = 3
        B_CR = 4
        A = 5
        B = 6
        SR = 7
        CR = 8
        UNDEFINED = 99

        @staticmethod
        def from_string(category_str: str) -> DataSet.DataSetCategory:
            key_map = {
                DataSet.DataSetCategory.A_SR: ("a", "sr"),
                DataSet.DataSetCategory.A_CR: ("a", "cr"),
                DataSet.DataSetCategory.B_SR: ("b", "sr"),
                DataSet.DataSetCategory.B_CR: ("b", "cr"),
            }
            for category, strings in key_map.items():
                if all(s in re.split(r"[_\- ]", category_str.lower()) for s in strings):
                    return category
            return DataSet.DataSetCategory.UNDEFINED

        def __add__(self, other: DataSet.DataSetCategory) -> DataSet.DataSetCategory:
            if self == other:
                return self
            if (self in [DataSet.DataSetCategory.A_SR, DataSet.DataSetCategory.A_CR] and
                    other in [DataSet.DataSetCategory.A_SR, DataSet.DataSetCategory.A_CR]):
                return DataSet.DataSetCategory.A
            if (self in [DataSet.DataSetCategory.B_SR, DataSet.DataSetCategory.B_CR] and
                    other in [DataSet.DataSetCategory.B_SR, DataSet.DataSetCategory.B_CR]):
                return DataSet.DataSetCategory.B
            if (self in [DataSet.DataSetCategory.A_SR, DataSet.DataSetCategory.B_SR] and
                    other in [DataSet.DataSetCategory.A_SR, DataSet.DataSetCategory.B_SR]):
                return DataSet.DataSetCategory.SR
            if (self in [DataSet.DataSetCategory.A_CR, DataSet.DataSetCategory.B_CR] and
                    other in [DataSet.DataSetCategory.A_CR, DataSet.DataSetCategory.B_CR]):
                return DataSet.DataSetCategory.CR
            return DataSet.DataSetCategory.UNDEFINED

    def __init__(
            self,
            data: Optional[Union[npt.NDArray, pd.DataFrame]] = None,
            observable_names: Optional[List[str]] = None,
            category: DataSetCategory = DataSetCategory.UNDEFINED,
        ):
        self._category = category
        if data is None:
            self._data = pd.DataFrame()
        elif isinstance(data, np.ndarray):
            if data.ndim == 1 and len(data) == 0:
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
        
    def __add__(self, other: DataSet) -> DataSet:
        if self.empty:
            return other
        if other.empty:
            return self
        if self.observable_names != other.observable_names:
            raise ValueError("Observable names do not match between datasets.")
        
        _data = pd.concat((self._data, other._data), axis=0)
        _data.reset_index(level=0, drop=True, inplace=True)
        category = self.category + other.category

        return DataSet(data=_data, observable_names=self.observable_names, category=category)
    
    def __mul__(self, other: ShiftAndNormalizationFactor) -> DataSet:
        assert isinstance(other, ShiftAndNormalizationFactor), \
            f"Dataset multiplication is only allowed by a ShiftAndNormalizationFactor, not {type(other)}"

        result = self.create_copy()        
        for obs in self.observable_names:
            try:
                native_offset = min(result._data[obs])
                result._data[obs] -= native_offset
                result._data[obs] *= other.get_factor(obs)
                result._data[obs] += other.get_offset(obs) + native_offset
            except KeyError:
                raise ArithmeticError(f"No factor for observable {obs} in multiplication")
            
        return result

    def __rmul__(self, other: ShiftAndNormalizationFactor) -> DataSet:
        return self.__mul__(other)

    def __truediv__(self, other: ShiftAndNormalizationFactor) -> DataSet:
        assert isinstance(other, ShiftAndNormalizationFactor), \
            f"Dataset division is only allowed by a ShiftAndNormalizationFactor, not {type(other)}"
        
        result = self.create_copy()
        for obs in self.observable_names:
            try:
                native_offset = min(result._data[obs])
                result._data[obs] -= native_offset
                result._data[obs] /= other.get_factor(obs)
                result._data[obs] -= other.get_offset(obs) - native_offset
            except KeyError:
                raise ArithmeticError(f"No factor for observable {obs} in division")
            
        return result

    def __getitem__(self, item: Union[int, slice, npt.NDArray]) -> DataSet:
        return DataSet(
            data=pd.DataFrame(self._data.iloc[item, :]),
            observable_names=self.observable_names,
            category=self._category,
        )

    def create_copy(self) -> DataSet:
        return DataSet(data=self._data.copy(), observable_names=self.observable_names, category=self._category)

    @property
    def category(self) -> DataSetCategory:
        return self._category
    
    @category.setter
    def category(self, new_category: DataSetCategory):
        self._category = new_category

    def __radd__(self, other: DataSet) -> DataSet:
        return self.__add__(other)

    @property
    def observable_names(self) -> List[str]:
        return self._data.columns.tolist()

    @observable_names.setter
    def observable_names(self, names: Iterable[str]):
        self._data.columns = list(names)

    @property
    def n_observables(self) -> int:
        return len(self.observable_names)

    @property
    def n_samples(self):
        return self._data.shape[0]

    @property
    def empty(self) -> bool:
        return self.n_samples == 0

    @property
    def events(self) -> npt.NDArray:
        return self._data.to_numpy()

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
    
    def get_normalized(self) -> Tuple[DataSet, ShiftAndNormalizationFactor]:
        offsets = {}
        factors = {}
        result = self.create_copy()
        for obs in result.observable_names:
            obs_slice = result.slice_along_observable_names(obs)

            # shift and scale to fit range [-1, 1]
            minimum = np.min(obs_slice)
            span = np.ptp(obs_slice)
            if span == 0:
                # A constant observable carries no variation for the model.
                # Keep its transform finite and map it to zero.
                offsets[obs] = minimum
                factors[obs] = 1.0
            else:
                offsets[obs] = minimum + 1
                factors[obs] = span / 2

        normalization_factor = ShiftAndNormalizationFactor(factors, offsets)
        return result / normalization_factor, normalization_factor

    def filter(self, filter: np.ndarray) -> DataSet:
        """
        Filter the dataset according to a boolean mask.
        """
        filtered_data = self._data.iloc[filter, :]
        return DataSet(data=filtered_data, observable_names=self.observable_names, category=self._category)

    def filter_observable_names(self, observables: Union[str, List[str]]) -> DataSet:
        return DataSet(
            data=self.slice_along_observable_names(observables),
            observable_names=[observables] if isinstance(observables, str) else observables,
            category=self._category,
        )


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

    idx = np.random.choice(
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


@dataclass
class ShiftAndNormalizationFactor:
    """
    Normalization factor to be applied to datasets for training and fitting.
    """
    _factors: Dict[str, float]
    _offsets: Dict[str, float]

    def __post_init__(self, **kwargs):
        assert all(key in self._offsets for key in self._factors)

    @property
    def n_dim(self) -> int:
        return len(self._factors)
    
    def get_offset(self, key: str) -> float:
        return self._offsets[key]

    def get_factor(self, key: str) -> float:
        return self._factors[key]
