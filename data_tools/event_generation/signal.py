from typing import Optional

import numpy as np
from scipy import stats

from data_tools.data_utils import DataSet
from data_tools.event_generation.distribution import DataDistribution
from data_tools.event_generation.types import FLOAT_OR_ARRAY

_GAUSSIAN_INTEGRATION_STANDARD_DEVIATIONS = 6.0

# Namespace for signal generating functions
# classes defined here that inherit from DataDistribution
# are automatically recognized by the program and can be
# called from the config file by snake case class name.

class SignalDistribution(DataDistribution):
    """Base class for distributions defined in the signal namespace."""


class NoSignal(SignalDistribution):
    """
    No signal distribution.
    """

    def generate_amount(
        self,
        amount: int,
    ) -> DataSet:
        return DataSet(np.empty(shape=(0, self._number_of_dimensions)))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        return np.zeros_like(x)


class GaussianSignal(SignalDistribution):
    
    def __init__(
        self,
        number_of_dimensions: int,
        location: float,
        gaussian_signal_sigma: float,
        domain_max: Optional[float] = None,
    ):
        integration_upper_limit = (
            location
            + _GAUSSIAN_INTEGRATION_STANDARD_DEVIATIONS * gaussian_signal_sigma
            if domain_max is None
            else domain_max
        )
        if not np.isfinite(integration_upper_limit) or integration_upper_limit <= 0:
            raise ValueError("domain_max must define a positive finite upper limit.")
        super().__init__(number_of_dimensions, domain_max=integration_upper_limit)
        self._location = location
        self._gaussian_signal_sigma = gaussian_signal_sigma

    def generate_amount(
        self,
        amount: int,
    ) -> DataSet:
        return DataSet(np.random.normal(
            loc=self._location,
            scale=self._gaussian_signal_sigma,
            size=(amount, self._number_of_dimensions)
        ))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        sigma, mean = self._gaussian_signal_sigma, self._location
        values = np.asarray(x)
        densities = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(
            -(values - mean) ** 2 / (2 * sigma**2)
        )
        if self._number_of_dimensions == 1:
            return densities
        if values.ndim == 0:
            raise ValueError(
                "An N-D Gaussian PDF requires one coordinate per dimension."
            )
        if values.shape[-1] != self._number_of_dimensions:
            raise ValueError(
                f"Expected {self._number_of_dimensions} coordinates, got "
                f"{values.shape[-1]}."
            )
        return np.prod(densities, axis=-1)


class MultivariateGaussianSignal(SignalDistribution):
    """A joint Gaussian signal that can correlate observable coordinates."""

    def __init__(
        self,
        number_of_dimensions: int,
        mean: list,
        covariance: list,
        domain_max: Optional[float] = None,
    ):
        self._mean = np.asarray(mean, dtype=float)
        self._covariance = np.asarray(covariance, dtype=float)

        expected_mean_shape = (number_of_dimensions,)
        expected_covariance_shape = (number_of_dimensions, number_of_dimensions)
        if self._mean.shape != expected_mean_shape:
            raise ValueError(
                f"Multivariate Gaussian mean must have shape {expected_mean_shape}, "
                f"got {self._mean.shape}."
            )
        if self._covariance.shape != expected_covariance_shape:
            raise ValueError(
                "Multivariate Gaussian covariance must have shape "
                f"{expected_covariance_shape}, got {self._covariance.shape}."
            )
        if not np.all(np.isfinite(self._mean)) or not np.all(
            np.isfinite(self._covariance)
        ):
            raise ValueError("Multivariate Gaussian parameters must be finite.")
        if not np.allclose(self._covariance, self._covariance.T):
            raise ValueError("Multivariate Gaussian covariance must be symmetric.")
        if np.min(np.linalg.eigvalsh(self._covariance)) < -1e-10:
            raise ValueError(
                "Multivariate Gaussian covariance must be positive semidefinite."
            )

        marginal_standard_deviations = np.sqrt(np.diag(self._covariance))
        self._integration_upper_limits = (
            self._mean
            + _GAUSSIAN_INTEGRATION_STANDARD_DEVIATIONS
            * marginal_standard_deviations
            if domain_max is None
            else np.full(number_of_dimensions, domain_max, dtype=float)
        )
        if not np.all(np.isfinite(self._integration_upper_limits)) or np.any(
            self._integration_upper_limits <= 0
        ):
            raise ValueError("domain_max must define positive finite upper limits.")
        super().__init__(
            number_of_dimensions,
            domain_max=float(np.max(self._integration_upper_limits)),
        )

        self._frozen_distribution = stats.multivariate_normal(
            mean=self._mean,
            cov=self._covariance,
            allow_singular=True,
        )

    @property
    def integration_upper_limits(self) -> np.ndarray:
        return self._integration_upper_limits.copy()

    def generate_amount(self, amount: int) -> DataSet:
        return DataSet(np.random.multivariate_normal(
            mean=self._mean,
            cov=self._covariance,
            size=amount,
            check_valid="raise",
        ))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        return self._frozen_distribution.pdf(x)


class NonlocalSignal(SignalDistribution):

    def generate_amount(
        self,
        amount: int,
    ) -> DataSet:
        return super().generate_amount(amount)

    def __init__(
        self,
        number_of_dimensions: int,
        param_scale: float = 1,
        domain_min: float = 0,
        domain_max: float = 1e2,
        domain_granularity: int = 100000,
    ):
        super().__init__(
            number_of_dimensions,
            domain_min=domain_min,
            domain_max=domain_max,
            domain_granularity=domain_granularity,
        )
        self._param_scale = param_scale

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        x = x * self._param_scale
        dist = x**2 * np.exp(-x)
        normalization = 2  # Definite integral in [0, inf) is 2
        jacobian = self._param_scale
        return dist / normalization * jacobian
