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
            parts = re.split(r"[_\- ]", category_str.lower())
            if "a" in parts and "sr" in parts:
                return DATASET_REGIONS.sr.a
            if "a" in parts and "cr" in parts:
                return DATASET_REGIONS.cr.a
            if "b" in parts and "sr" in parts:
                return DATASET_REGIONS.sr.b
            if "b" in parts and "cr" in parts:
                return DATASET_REGIONS.cr.b
            return DataSet.DataSetCategory.UNDEFINED

        def __add__(self, other: DataSet.DataSetCategory) -> DataSet.DataSetCategory:
            if self == other:
                return self
            if DATASET_REGIONS.is_a(self) and DATASET_REGIONS.is_a(other):
                return DataSet.DataSetCategory.A
            if DATASET_REGIONS.is_b(self) and DATASET_REGIONS.is_b(other):
                return DataSet.DataSetCategory.B
            if DATASET_REGIONS.sr.contains(self) and DATASET_REGIONS.sr.contains(other):
                return DataSet.DataSetCategory.SR
            if DATASET_REGIONS.cr.contains(self) and DATASET_REGIONS.cr.contains(other):
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
                result._data[obs] = other.denormalize_values(
                    result._data[obs].to_numpy(), (obs,)
                )
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
                result._data[obs] = other.normalize_values(
                    result._data[obs].to_numpy(), (obs,)
                )
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
                offsets[obs] = minimum - 1
                factors[obs] = 1.0
            else:
                offsets[obs] = minimum
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


@dataclass(frozen=True)
class DataSetRegion:
    """The A and B categories of one physical region."""

    a: DataSet.DataSetCategory
    b: DataSet.DataSetCategory

    def contains(self, category: DataSet.DataSetCategory) -> bool:
        return category == self.a or category == self.b


@dataclass(frozen=True)
class DataSetRegions:
    """The single category topology used by dataset loading and batching."""

    sr: DataSetRegion
    cr: DataSetRegion

    def for_category(self, category: DataSet.DataSetCategory) -> DataSetRegion:
        if self.sr.contains(category):
            return self.sr
        if self.cr.contains(category):
            return self.cr
        raise KeyError(f"Dataset category '{category}' does not belong to an SR or CR pair.")

    def is_a(self, category: DataSet.DataSetCategory) -> bool:
        return category == self.sr.a or category == self.cr.a

    def is_b(self, category: DataSet.DataSetCategory) -> bool:
        return category == self.sr.b or category == self.cr.b


DATASET_REGIONS = DataSetRegions(
    sr=DataSetRegion(
        a=DataSet.DataSetCategory.A_SR,
        b=DataSet.DataSetCategory.B_SR,
    ),
    cr=DataSetRegion(
        a=DataSet.DataSetCategory.A_CR,
        b=DataSet.DataSetCategory.B_CR,
    ),
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
    """One explicit affine map shared by data and feature geometry.

    For every nonconstant observable, ``normalize_values`` applies
    ``(x - offset) / factor - 1`` and ``denormalize_values`` is its exact
    inverse.  Constant observables use a finite, deliberately shifted offset
    so their sole value maps to zero.
    """
    _factors: Dict[str, float]
    _offsets: Dict[str, float]

    def __post_init__(self, **kwargs):
        assert set(self._factors) == set(self._offsets)
        assert all(np.isfinite(value) and value > 0 for value in self._factors.values())
        assert all(np.isfinite(value) for value in self._offsets.values())

    @property
    def n_dim(self) -> int:
        return len(self._factors)
    
    def get_offset(self, key: str) -> float:
        return self._offsets[key]

    def get_factor(self, key: str) -> float:
        return self._factors[key]

    def normalize_values(
        self,
        values: npt.ArrayLike,
        observable_names: Iterable[str],
    ) -> npt.NDArray[np.float64]:
        """Apply this factor's affine map along the observable axis."""

        array, offsets, factors = self._affine_parameters(values, observable_names)
        return (array - offsets) / factors - 1

    def denormalize_values(
        self,
        values: npt.ArrayLike,
        observable_names: Iterable[str],
    ) -> npt.NDArray[np.float64]:
        """Invert :meth:`normalize_values` along the observable axis."""

        array, offsets, factors = self._affine_parameters(values, observable_names)
        return (array + 1) * factors + offsets

    def scale_values(
        self,
        values: npt.ArrayLike,
        observable_names: Iterable[str],
    ) -> npt.NDArray[np.float64]:
        """Express physical coordinate widths in normalized units."""

        array, _, factors = self._affine_parameters(values, observable_names)
        return array / factors

    def _affine_parameters(
        self,
        values: npt.ArrayLike,
        observable_names: Iterable[str],
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64] | float,
        npt.NDArray[np.float64] | float,
    ]:
        """Return validated values and broadcastable parameters for one coordinate map."""

        array = np.asarray(values, dtype=float)
        names = tuple(observable_names)
        if not names:
            raise ValueError("At least one observable name is required.")
        if array.ndim == 1 and len(names) == 1:
            return array, self.get_offset(names[0]), self.get_factor(names[0])
        if array.ndim >= 1 and array.shape[-1] == len(names):
            return (
                array,
                np.asarray([self.get_offset(name) for name in names]),
                np.asarray([self.get_factor(name) for name in names]),
            )
        raise ValueError("Values must have one final coordinate per observable.")
