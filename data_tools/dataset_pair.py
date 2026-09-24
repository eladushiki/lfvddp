"""Regional A/B dataset pairing and post-materialization split policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from data_tools.data_utils import DataSet

if TYPE_CHECKING:
    from data_tools.dataset_config import DatasetParameters


class DatasetPairSplitPolicy(ABC):
    """Define how fully materialized A/B datasets are finalized."""

    @abstractmethod
    def split(
        self,
        a: DataSet,
        b: DataSet,
    ) -> Tuple[DataSet, DataSet]:
        """Return finalized A and B datasets for one physical region."""


@dataclass(frozen=True)
class IdentityDatasetPairSplitPolicy(DatasetPairSplitPolicy):
    """Keep separately materialized A and B datasets unchanged."""

    def split(
        self,
        a: DataSet,
        b: DataSet,
    ) -> Tuple[DataSet, DataSet]:
        return a, b


@dataclass(frozen=True)
class ShuffledDatasetPairSplitPolicy(DatasetPairSplitPolicy):
    """Shuffle a regional A/B pool, then restore its original split sizes."""

    replacement: bool

    def split(
        self,
        a: DataSet,
        b: DataSet,
    ) -> Tuple[DataSet, DataSet]:
        a_size = a.n_samples
        shuffled = (a + b)[np.random.permutation(a_size + b.n_samples)]
        a_result = shuffled[:a_size]
        b_result = shuffled[a_size:]
        a_result.category = a.category
        b_result.category = b.category
        return a_result, b_result


@dataclass(frozen=True)
class RegionalDataPair:
    """The two datasets and common finalization policy of one SR or CR region."""

    a: DataSet
    a_parameters: DatasetParameters
    b: DataSet
    b_parameters: DatasetParameters
    split_policy: DatasetPairSplitPolicy

    def finalized(self) -> RegionalDataPair:
        a, b = self.split_policy.split(self.a, self.b)
        return RegionalDataPair(
            a=a,
            a_parameters=self.a_parameters,
            b=b,
            b_parameters=self.b_parameters,
            split_policy=IdentityDatasetPairSplitPolicy(),
        )

    def get(self, category: DataSet.DataSetCategory) -> Tuple[DataSet, DatasetParameters]:
        if category == self.a.category:
            return self.a, self.a_parameters
        if category == self.b.category:
            return self.b, self.b_parameters
        raise KeyError(f"Dataset category '{category}' is not part of this regional pair.")
