from data_tools.data_utils import DataSet
from data_tools.event_generation.distribution import DataDistribution
from data_tools.event_generation.types import FLOAT_OR_ARRAY
import numpy as np

# Namespace for signal generating functions
# classes defined here that inherit from DataDistribution
# are automatically recognized by the program and can be
# called from the config file by snake case class name.

class SignalDistribution(DataDistribution):
    def __init__(self, number_of_dimensions: int, location: float):
        super().__init__(number_of_dimensions)
        self._location = location


class NoSignal(SignalDistribution):
    """
    No signal distribution.
    """

    def __init__(self, number_of_dimensions: int):
        super().__init__(number_of_dimensions, location=0)

    def generate_amount(
        self,
        amount: int,
        domain_min: float = 0,
        domain_max: float = 1e2,
        domain_granularity: int = 100000,
    ) -> DataSet:
        return DataSet(np.empty(shape=(0, self._number_of_dimensions)))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        return np.zeros_like(x)


class GaussianSignal(SignalDistribution):
    
    def __init__(self, number_of_dimensions: int, location: float, gaussian_signal_sigma: float):
        super().__init__(number_of_dimensions, location)
        self._gaussian_signal_sigma = gaussian_signal_sigma

    def generate_amount(
        self,
        amount: int,
        domain_min: float = 0,
        domain_max: float = 1e2,
        domain_granularity: int = 100000,
    ) -> DataSet:
        return DataSet(np.random.normal(
            loc=self._location,
            scale=self._gaussian_signal_sigma,
            size=(amount, self._number_of_dimensions)
        ))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        sigma, mean = self._gaussian_signal_sigma, self._location
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * sigma**2))


class NonlocalSignal(SignalDistribution):

    def generate_amount(
        self,
        amount: int,
        domain_min: float = 0,
        domain_max: float = 1e2,
        domain_granularity: int = 100000,
    ) -> DataSet:
        return super().generate_amount(amount, domain_min, domain_max, domain_granularity)

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        dist = x**2 * np.exp(-x)
        normalization = 2  # Definite integral in [0, inf) is 2
        return dist / normalization
