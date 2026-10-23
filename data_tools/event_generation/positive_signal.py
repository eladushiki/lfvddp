"""Condition a signal on the nonnegative observable region."""

from functools import cached_property

import numpy as np

from data_tools.data_utils import DataSet
from data_tools.event_generation.distribution import (
    DataDistribution,
    validate_generated_dataset,
)
from data_tools.event_generation.signal import NoSignal
from data_tools.event_generation.types import FLOAT_OR_ARRAY


def condition_signal_distribution(
    distribution: DataDistribution,
    number_of_dimensions: int,
    positive_only: bool,
) -> DataDistribution:
    """Apply the shared signal-domain policy to sampling and calibration."""
    if positive_only and not isinstance(distribution, NoSignal):
        return PositiveSignalDistribution(distribution, number_of_dimensions)
    return distribution


class PositiveSignalDistribution(DataDistribution):
    """Reject whole events, preserving the joint conditional distribution."""

    def __init__(self, distribution: DataDistribution, number_of_dimensions: int):
        super().__init__(number_of_dimensions)
        self._distribution = distribution

    @property
    def integration_upper_limits(self) -> np.ndarray:
        return self._distribution.integration_upper_limits

    @cached_property
    def positive_region_probability(self) -> float:
        probability = self._distribution.positive_region_probability
        if not np.isfinite(probability) or not 0 < probability <= 1:
            raise ValueError("Signal has no finite positive-region probability.")
        return probability

    def generate_amount(self, amount: int) -> DataSet:
        if amount == 0:
            return DataSet(np.empty((0, self._number_of_dimensions)))
        probability = self.positive_region_probability
        remaining = amount
        accepted = []
        # Bound rejection sampling so impossible or extremely rare regions fail
        # explicitly instead of hanging or silently returning too few events.
        budget = max(1_000_000, 100 * amount)
        while remaining and budget:
            batch_size = min(budget, 100_000)
            if probability * batch_size > 1.1 * remaining:
                batch_size = max(remaining, int(np.ceil(1.1 * remaining / probability)))
            batch = self._distribution.generate_amount(batch_size)
            validate_generated_dataset(
                batch, batch_size, self._number_of_dimensions, "signal"
            )
            events = batch.events
            selected = events[np.all(np.isfinite(events) & (events >= 0), axis=1)]
            selected = selected[:remaining]
            accepted.append(selected)
            remaining -= len(selected)
            budget -= batch_size
        if remaining:
            raise ValueError(
                "Positive-only signal sampling exhausted its draw budget; "
                "choose parameters with greater positive-region probability "
                "or set dataset__signal_positive_only=false."
            )
        return DataSet(np.concatenate(accepted))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        values = np.asarray(x)
        inside = values >= 0
        if self._number_of_dimensions > 1:
            inside = np.all(inside, axis=-1)
        return np.where(
            inside, self._distribution.pdf(x) / self.positive_region_probability, 0.0
        )
