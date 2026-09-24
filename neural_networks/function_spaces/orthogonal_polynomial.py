"""Orthogonal-polynomial function-space family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch

from data_tools.data_utils import ShiftAndNormalizationFactor

from neural_networks.function_spaces.base import (
    DeterministicFeatureFunction,
    EventInput,
    dimensions,
    events_tensor,
    require_options,
)
from frame.value_enum import ValueEnum


class PolynomialBasis(ValueEnum):
    LEGENDRE = "legendre"
    CHEBYSHEV = "chebyshev"

@dataclass(frozen=True)
class OrthogonalPolynomialGeometry:
    basis: PolynomialBasis
    maximum_degree: int
    domain: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if self.maximum_degree < 0:
            raise ValueError("maximum_degree must be non-negative.")
        if not self.domain or any(minimum >= maximum for minimum, maximum in self.domain):
            raise ValueError("Each polynomial domain must contain an increasing (minimum, maximum) pair.")

    @property
    def feature_count(self) -> int:
        return len(self.domain) * (self.maximum_degree + 1)

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> "OrthogonalPolynomialGeometry":
        require_options(options, "orthogonal_polynomial", ("basis", "maximum_degree", "domain"))
        domains = dimensions(options["domain"], "domain")
        if any(len(domain) != 2 or domain[0] >= domain[1] for domain in domains):
            raise ValueError("Each polynomial domain must contain an increasing (minimum, maximum) pair.")
        return cls(
            PolynomialBasis.parse(options["basis"]),
            int(options["maximum_degree"]),
            tuple((domain[0], domain[1]) for domain in domains),
        )


class OrthogonalPolynomialFunction(DeterministicFeatureFunction):
    """Legendre or Chebyshev additive polynomial feature map."""

    family = "orthogonal_polynomial"

    def __init__(
        self,
        geometry: OrthogonalPolynomialGeometry,
        *,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: Optional[torch.device | str] = None,
        output_dimension: int = 1,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.geometry = geometry
        self._input_dimension = len(geometry.domain)
        super().__init__(
            geometry.feature_count,
            output_dimension=output_dimension,
            dtype=dtype,
            device=device,
            options=options,
        )
        self.register_buffer("_domain", torch.tensor(geometry.domain, dtype=dtype, device=device))

    @classmethod
    def geometry_from_options(
        cls, options: Mapping[str, Any]
    ) -> OrthogonalPolynomialGeometry:
        return OrthogonalPolynomialGeometry.from_options(options)

    def normalize_input_geometry(
        self,
        normalization_factor: ShiftAndNormalizationFactor,
        observable_names: tuple[str, ...],
    ) -> None:
        """Derive a normalized polynomial domain from physical configuration values."""

        if len(observable_names) != self.input_dimension:
            raise ValueError("Function-space geometry does not match the observable dimension.")
        normalized_domain = normalization_factor.normalize_values(
            np.asarray(self.geometry.domain).T, observable_names
        ).T
        with torch.no_grad():
            self._domain.copy_(torch.as_tensor(normalized_domain, dtype=self._domain.dtype, device=self._domain.device))

    @classmethod
    def _statistical_constraint_dimension_for_geometry(
        cls, geometry: OrthogonalPolynomialGeometry
    ) -> int:
        del cls
        return len(geometry.domain)

    def features(self, events: EventInput) -> torch.Tensor:
        values = events_tensor(
            events,
            self.input_dimension,
            dtype=self.coefficients.dtype,
            device=self.coefficients.device,
        )
        minimum, maximum = self._domain[:, 0], self._domain[:, 1]
        normalized = 2 * (values - minimum) / (maximum - minimum) - 1
        columns = []
        for dimension in range(self.input_dimension):
            current = torch.ones_like(normalized[:, dimension])
            columns.append(current)
            if self.geometry.maximum_degree >= 1:
                previous = current
                current = normalized[:, dimension]
                columns.append(current)
                for degree in range(2, self.geometry.maximum_degree + 1):
                    if self.geometry.basis is PolynomialBasis.LEGENDRE:
                        next_value = (
                            (2 * degree - 1) * normalized[:, dimension] * current
                            - (degree - 1) * previous
                        ) / degree
                    else:
                        next_value = 2 * normalized[:, dimension] * current - previous
                    columns.append(next_value)
                    previous, current = current, next_value
        return torch.stack(columns, dim=1)
