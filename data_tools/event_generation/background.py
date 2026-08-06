from data_tools.data_utils import DataSet
from data_tools.event_generation.distribution import DataDistribution
from data_tools.event_generation.types import FLOAT_OR_ARRAY
import numpy as np
from scipy import stats

# Namespace for background generating functions
# classes defined here that inherit from DataDistribution
# are automatically recognized by the program and can be
# called from the config file by snake case class name.


class ExponentialBackground(DataDistribution):

    def generate_amount(self, amount: int) -> DataSet:
        return DataSet(np.random.exponential(
            size=(amount, self._number_of_dimensions),
        ))
    
    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        values = np.asarray(x)
        densities = np.exp(-values)
        if self._number_of_dimensions == 1:
            return densities
        if values.ndim == 0:
            raise ValueError(
                "An N-D exponential PDF requires one coordinate per dimension."
            )
        if values.shape[-1] != self._number_of_dimensions:
            raise ValueError(
                f"Expected {self._number_of_dimensions} coordinates, got "
                f"{values.shape[-1]}."
            )
        return np.prod(densities, axis=-1)


class GaussianBackground(DataDistribution):

    def __init__(
        self,
        number_of_dimensions: int,
        domain_min: float = 0,
        domain_max: float = 100,
        mean: float = 0,
    ):
        """Independent unit-width truncated Gaussians in every dimension."""
        super().__init__(
            number_of_dimensions,
            domain_min=domain_min,
            domain_max=domain_max,
        )
        self._mean = mean
        self._covariance_matrix = np.eye(number_of_dimensions)
  
    def generate_amount(self, amount: int) -> DataSet:
        samples = []
        for dim in range(self._number_of_dimensions):
            # Create truncated normal distribution for each dimension
            std_dev = np.sqrt(self._covariance_matrix[dim, dim])
            a = (self._domain_min - self._mean) / std_dev
            b = (self._domain_max - self._mean) / std_dev
            
            # Generate samples using truncated normal
            dim_samples = stats.truncnorm.rvs(
                a=a, b=b,
                loc=self._mean,
                scale=self._covariance_matrix[dim, dim]**0.5,
                size=amount,
            )
            
            samples.append(dim_samples)
        
        return DataSet(np.column_stack(samples))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        if self._number_of_dimensions == 1:
            return stats.norm(loc=self._mean, scale=1).pdf(x)
        return stats.multivariate_normal(
            mean=np.full(self._number_of_dimensions, self._mean),
            cov=self._covariance_matrix,
        ).pdf(x)
