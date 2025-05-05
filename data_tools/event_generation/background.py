from data_tools.data_utils import DataSet
from data_tools.event_generation.distribution import DataDistribution
from data_tools.event_generation.types import FLOAT_OR_ARRAY
import numpy as np

# Namespace for background generating functions
# classes defined here that inherit from DataDistribution
# are automatically recognized by the program and can be
# called from the config file by snake case class name.


class ExponentialBackground(DataDistribution):

    def generate_amount(
            self, amount: int,
            domain_min: float = 0,
            domain_max: float = 100,
            domain_granularity: int = 100000
    ) -> DataSet:
        return DataSet(np.random.exponential(
            size=(amount, self._number_of_dimensions),
        ))
    
    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        return np.exp(-x)


class GaussianBackground(DataDistribution):
    
    def __init__(self, number_of_dimensions: int):
        """
        Mean is 0 and covariance matrix is a matrix of ones.
        """
        super().__init__(number_of_dimensions)
        self._covariance_matrix = np.ones((number_of_dimensions, number_of_dimensions))
  
    def generate_amount(
            self, amount: int,
            domain_min: float = 0,
            domain_max: float = 100,
            domain_granularity: int = 100000
    ) -> DataSet:
        return DataSet(np.random.multivariate_normal(
            mean=np.zeros(self._number_of_dimensions),
            cov=self._covariance_matrix,
            size=amount,
        ))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        k = self._number_of_dimensions
        det_std = np.linalg.det(self._covariance_matrix)
        return (2 * np.pi)**(-k / 2) * det_std**(-1 / 2) * \
                np.exp(-0.5 * np.dot(x, np.dot(np.linalg.inv(self._covariance_matrix), x)))
