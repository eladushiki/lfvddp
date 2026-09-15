"""Shared fixed-centre feature geometry for sigmoid and radial-basis families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch

from neural_networks.function_spaces.base import (
    DeterministicFeatureFunction,
    EventInput,
    events_tensor,
    number_sequence,
    require_options,
)


@dataclass(frozen=True)
class CenterGeometry:
    centers: tuple[tuple[float, ...], ...]
    widths: tuple[tuple[float, ...], ...]

    @property
    def feature_count(self) -> int:
        return len(self.centers)

    @property
    def input_dimension(self) -> int:
        return len(self.centers[0])

    @classmethod
    def from_options(cls, options: Mapping[str, Any], family_name: str) -> "CenterGeometry":
        require_options(options, family_name, ("centers", "widths"))
        raw_centers = options["centers"]
        if isinstance(raw_centers, (str, bytes)):
            raise ValueError(f"{family_name} centers must be numeric geometry.")
        center_array = np.asarray(raw_centers, dtype=object)
        if center_array.ndim == 0:
            centers = ((float(raw_centers),),)
        elif center_array.ndim == 1:
            centers = tuple((float(center),) for center in raw_centers)
        elif center_array.ndim == 2:
            centers = tuple(number_sequence(center, "centers") for center in raw_centers)
        else:
            raise ValueError(f"{family_name} centers must be one- or two-dimensional.")
        if not centers or any(not all(np.isfinite(value) for value in center) for center in centers):
            raise ValueError(f"{family_name} centers must contain at least one finite value.")
        dimension = len(centers[0])
        if any(len(center) != dimension for center in centers):
            raise ValueError(f"{family_name} centers must have consistent dimensionality.")

        raw_widths = options["widths"]
        if isinstance(raw_widths, (str, bytes)):
            raise ValueError(f"{family_name} widths must be numeric geometry.")
        width_array = np.asarray(raw_widths, dtype=object)
        if width_array.ndim == 0:
            width_matrix = tuple((float(raw_widths),) * dimension for _ in centers)
        elif width_array.ndim == 1:
            width_values = number_sequence(raw_widths, "widths")
            if len(width_values) == 1:
                width_matrix = tuple(width_values * dimension for _ in centers)
            elif len(width_values) == len(centers):
                width_matrix = tuple((width,) * dimension for width in width_values)
            elif len(centers) == 1 and len(width_values) == dimension:
                width_matrix = (width_values,)
            else:
                raise ValueError(
                    f"{family_name} widths must be scalar, one per center, or one vector per center."
                )
        elif width_array.ndim == 2 and len(width_array) == len(centers):
            width_matrix = tuple(number_sequence(width, "widths") for width in raw_widths)
            if any(len(width) != dimension for width in width_matrix):
                raise ValueError(f"{family_name} widths must match the center dimensionality.")
        else:
            raise ValueError(
                f"{family_name} widths must be scalar, one per center, or one vector per center."
            )
        if any(width <= 0 for widths in width_matrix for width in widths):
            raise ValueError(f"{family_name} widths must be strictly positive.")
        return cls(tuple(centers), tuple(width_matrix))


class CenteredFeatureFunction(DeterministicFeatureFunction):
    """Base class that owns centre tensors and normalized input conversion."""

    def __init__(
        self,
        geometry: CenterGeometry,
        *,
        dtype: torch.dtype = torch.get_default_dtype(),
        device: Optional[torch.device | str] = None,
        output_dimension: int = 1,
        options: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.geometry = geometry
        self._input_dimension = geometry.input_dimension
        super().__init__(
            geometry.feature_count,
            output_dimension=output_dimension,
            dtype=dtype,
            device=device,
            options=options,
        )
        self.register_buffer("_centers", torch.tensor(geometry.centers, dtype=dtype, device=device))
        self.register_buffer("_widths", torch.tensor(geometry.widths, dtype=dtype, device=device))

    def centered_values(self, events: EventInput) -> torch.Tensor:
        values = events_tensor(
            events,
            self.input_dimension,
            dtype=self.coefficients.dtype,
            device=self.coefficients.device,
        )
        return (values[:, None, :] - self._centers[None, :, :]) / self._widths[None, :, :]
