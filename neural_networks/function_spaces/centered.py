"""Shared fixed-centre feature geometry for sigmoid and radial-basis families."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Optional

import numpy as np
import torch

from data_tools.data_utils import ShiftAndNormalizationFactor

from neural_networks.function_spaces.base import (
    DeterministicFeatureFunction,
    EventInput,
    dimensions,
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
        if not centers or any(
            not all(np.isfinite(value) for value in center) for center in centers
        ):
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
                    f"{family_name} widths must be scalar, one per center, "
                    "or one vector per center."
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
        if len(set(zip(centers, width_matrix))) != len(centers):
            raise ValueError(
                f"{family_name} must not define duplicate centre-and-width features."
            )
        return cls(tuple(centers), tuple(width_matrix))

    @classmethod
    def tensor_product_from_options(
        cls, options: Mapping[str, Any], family_name: str
    ) -> "CenterGeometry":
        """Expand per-dimension centre and width lists into joint features."""

        require_options(options, family_name, ("centers", "widths"))
        center_dimensions = dimensions(options["centers"], "centers")
        raw_widths = options["widths"]
        if np.isscalar(raw_widths):
            width_dimensions = tuple(
                (float(raw_widths),) * len(centers) for centers in center_dimensions
            )
        else:
            width_dimensions = dimensions(raw_widths, "widths")
            if len(width_dimensions) == 1 and len(center_dimensions) > 1:
                width_values = width_dimensions[0]
                if len(width_values) != len(center_dimensions):
                    raise ValueError(
                        f"{family_name} tensor-product widths must be scalar, "
                        "one per dimension, or match centers per dimension."
                    )
                width_dimensions = tuple(
                    (width,) * len(centers)
                    for width, centers in zip(width_values, center_dimensions)
                )
        if len(width_dimensions) != len(center_dimensions):
            raise ValueError(
                f"{family_name} tensor-product widths must match center dimensionality."
            )
        if any(
            len(widths) != len(centers)
            for widths, centers in zip(width_dimensions, center_dimensions)
        ):
            raise ValueError(
                f"{family_name} tensor-product widths must match centers per dimension."
            )
        if any(width <= 0 for widths in width_dimensions for width in widths):
            raise ValueError(f"{family_name} widths must be strictly positive.")

        features = tuple(
            (tuple(center for center, _ in items), tuple(width for _, width in items))
            for items in product(
                *tuple(
                    zip(centers, widths)
                    for centers, widths in zip(center_dimensions, width_dimensions)
                )
            )
        )
        centers = tuple(center for center, _ in features)
        widths = tuple(width for _, width in features)
        if len(set(zip(centers, widths))) != len(centers):
            raise ValueError(
                f"{family_name} must not define duplicate centre-and-width features."
            )
        return cls(centers, widths)


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
        self.register_buffer(
            "_centers", torch.tensor(geometry.centers, dtype=dtype, device=device)
        )
        self.register_buffer(
            "_widths", torch.tensor(geometry.widths, dtype=dtype, device=device)
        )

    def centered_values(self, events: EventInput) -> torch.Tensor:
        values = events_tensor(
            events,
            self.input_dimension,
            dtype=self.coefficients.dtype,
            device=self.coefficients.device,
        )
        return (values[:, None, :] - self._centers[None, :, :]) / self._widths[None, :, :]

    def normalize_input_geometry(
        self,
        normalization_factor: ShiftAndNormalizationFactor,
        observable_names: tuple[str, ...],
    ) -> None:
        """Derive normalized buffers from immutable physical centres and widths."""

        if len(observable_names) != self.input_dimension:
            raise ValueError("Function-space geometry does not match the observable dimension.")
        centers = normalization_factor.normalize_values(
            self.geometry.centers, observable_names
        )
        widths = normalization_factor.scale_values(self.geometry.widths, observable_names)
        with torch.no_grad():
            self._centers.copy_(
                torch.as_tensor(
                    centers, dtype=self._centers.dtype, device=self._centers.device
                )
            )
            self._widths.copy_(
                torch.as_tensor(
                    widths, dtype=self._widths.dtype, device=self._widths.device
                )
            )
