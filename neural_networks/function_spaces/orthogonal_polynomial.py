"""Orthogonal-polynomial function-space family."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

import torch

from neural_networks.function_spaces.base import (
    CoefficientTopology,
    DeterministicFeatureFunction,
    EventInput,
    FunctionSpaceMetadata,
    FunctionSpaceRegularity,
    dimensions,
    events_tensor,
    require_options,
)
from train.function_space_config import FunctionSpaceFamily


class PolynomialBasis(str, Enum):
    LEGENDRE = "legendre"
    CHEBYSHEV = "chebyshev"

    @classmethod
    def from_value(cls, value: "PolynomialBasis | str") -> "PolynomialBasis":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            raise ValueError("Polynomial basis must be 'legendre' or 'chebyshev'.") from error


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
            PolynomialBasis.from_value(options["basis"]),
            int(options["maximum_degree"]),
            tuple((domain[0], domain[1]) for domain in domains),
        )


class OrthogonalPolynomialFunction(DeterministicFeatureFunction):
    """Legendre or Chebyshev additive polynomial feature map."""

    family = FunctionSpaceFamily.ORTHOGONAL_POLYNOMIAL
    metadata = FunctionSpaceMetadata(
        FunctionSpaceRegularity.ORTHOGONAL_POLYNOMIAL,
        CoefficientTopology.LINEAR_COEFFICIENTS,
    )

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
    def from_options(
        cls,
        options: Mapping[str, Any],
        **construction: Any,
    ) -> "OrthogonalPolynomialFunction":
        return cls(
            OrthogonalPolynomialGeometry.from_options(options),
            options=options,
            **cls.construction_kwargs(options, construction),
        )

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
