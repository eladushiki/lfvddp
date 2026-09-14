"""Fixed-sigmoid function-space family."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from neural_networks.function_spaces.base import EventInput, FunctionSpaceMetadata
from neural_networks.function_spaces.centered import CenterGeometry, CenteredFeatureFunction
from train.function_space_config import FunctionSpaceFamily


class FixedSigmoidFunction(CenteredFeatureFunction):
    """Fixed-centre sigmoid features with trainable output coefficients only."""

    family = FunctionSpaceFamily.FIXED_SIGMOID
    metadata = FunctionSpaceMetadata("smooth", "linear_coefficients")

    @classmethod
    def from_options(cls, options: Mapping[str, Any], **construction: Any) -> "FixedSigmoidFunction":
        return cls(CenterGeometry.from_options(options, "fixed_sigmoid"), options=options, **construction)

    def features(self, events: EventInput) -> torch.Tensor:
        return torch.sigmoid(self.centered_values(events)).prod(dim=2)
