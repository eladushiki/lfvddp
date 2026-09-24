"""Regional A/B dataset pairing and post-materialization split policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Tuple

import numpy as np

from data_tools.data_utils import DataSet

if TYPE_CHECKING:
    from data_tools.dataset_config import DatasetParameters


REGIONAL_DATASET_CATEGORY_PAIRS = (
    (
        DataSet.DataSetCategory.A_SR,
        DataSet.DataSetCategory.B_SR,
    ),
    (
        DataSet.DataSetCategory.A_CR,
        DataSet.DataSetCategory.B_CR,
    ),
)


class DatasetPairSplitPolicy(ABC):
    """Define how fully materialized A/B datasets are finalized."""

    @abstractmethod
    def split(
        self,
        first: DataSet,
        second: DataSet,
    ) -> Tuple[DataSet, DataSet]:
        """Return finalized A and B datasets for one physical region."""


@dataclass(frozen=True)
class IdentityDatasetPairSplitPolicy(DatasetPairSplitPolicy):
    """Keep separately materialized A and B datasets unchanged."""

    def split(
        self,
        first: DataSet,
        second: DataSet,
    ) -> Tuple[DataSet, DataSet]:
        return first, second


@dataclass(frozen=True)
class ShuffledDatasetPairSplitPolicy(DatasetPairSplitPolicy):
    """Shuffle a regional A/B pool, then restore its original split sizes."""

    replacement: bool

    def split(
        self,
        first: DataSet,
        second: DataSet,
    ) -> Tuple[DataSet, DataSet]:
        first_size = first.n_samples
        shuffled = (first + second)[np.random.permutation(first_size + second.n_samples)]
        first_result = shuffled[:first_size]
        second_result = shuffled[first_size:]
        first_result.category = first.category
        second_result.category = second.category
        return first_result, second_result


@dataclass(frozen=True)
class RegionalDataPair:
    """The two datasets and common finalization policy of one SR or CR region."""

    first: Tuple[DataSet, DatasetParameters]
    second: Tuple[DataSet, DatasetParameters]
    split_policy: DatasetPairSplitPolicy

    def __post_init__(self) -> None:
        first_category = self.first[0].category
        second_category = self.second[0].category
        if (first_category, second_category) not in REGIONAL_DATASET_CATEGORY_PAIRS:
            raise ValueError(
                "A regional pair must contain A and B datasets from the same region, "
                f"got {first_category} and {second_category}."
            )

    def finalized(self) -> RegionalDataPair:
        first_dataset, second_dataset = self.split_policy.split(
            self.first[0],
            self.second[0],
        )
        return RegionalDataPair(
            first=(first_dataset, self.first[1]),
            second=(second_dataset, self.second[1]),
            split_policy=IdentityDatasetPairSplitPolicy(),
        )

    def __iter__(self) -> Iterator[Tuple[DataSet, DatasetParameters]]:
        yield self.first
        yield self.second

    def get(self, category: DataSet.DataSetCategory) -> Tuple[DataSet, DatasetParameters]:
        for dataset, parameters in self:
            if dataset.category == category:
                return dataset, parameters
        raise KeyError(f"Dataset category '{category}' is not part of this regional pair.")


def regional_categories_for(
    category: DataSet.DataSetCategory,
) -> Tuple[DataSet.DataSetCategory, DataSet.DataSetCategory]:
    """Return the A/B category pair that contains ``category``."""

    for regional_categories in REGIONAL_DATASET_CATEGORY_PAIRS:
        if category in regional_categories:
            return regional_categories
    raise KeyError(f"Dataset category '{category}' does not belong to an SR or CR pair.")
