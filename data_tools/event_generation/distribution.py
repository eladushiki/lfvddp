from abc import ABC, abstractmethod

from data_tools.data_utils import DataSet
from data_tools.event_generation.types import FLOAT_OR_ARRAY
import numpy as np


class DataDistribution(ABC):
    """
    A distribution with certain properties.

    When implementing a new distribution, the responsibility
    is on the developer to ensure the correctness and the
    normalization of the implemented functions.
    """

    def __init__(self,
                 number_of_dimensions: int,
    ):
        self._number_of_dimensions = number_of_dimensions

    @abstractmethod
    def generate_amount(
        self,
        amount: int,
        domain_min: float = 0,
        domain_max: float = 1e2,
        domain_granularity: int = 100000,
    ) -> DataSet:
        """
        Generate a sample of the distribution.

        This implementation draws from the pdf can be used in
        inherited classes, but need to be overriden explicitly.
        """
        rng = np.linspace(domain_min, domain_max, domain_granularity)

        probabilities = np.array([self.pdf(x) for x in rng])
        probabilities /= probabilities.sum()

        return DataSet(np.random.choice(
            rng,
            size=(amount, self._number_of_dimensions),
            replace=True,
            p=np.array([self.pdf(x) for x in probabilities]),
        ))

    @abstractmethod
    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        """
        Probability density function of the distribution.

        Should sum to 1.
        """
        pass
