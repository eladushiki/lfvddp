"""Canonical dimensions and parameter counts for adaptive neural networks."""

from itertools import pairwise
from typing import Any


def neural_architecture(
    input_dimension: Any,
    hidden_layer_nodes: Any,
    output_dimension: Any = 1,
) -> tuple[int, ...]:
    """Validate widths and expand the integer shorthand into a full architecture."""

    if type(input_dimension) is not int or input_dimension <= 0:
        raise ValueError("adaptive_neural requires a positive input_dimension.")
    if type(hidden_layer_nodes) is int:
        hidden_widths = (hidden_layer_nodes,)
    elif isinstance(hidden_layer_nodes, (list, tuple)):
        hidden_widths = tuple(hidden_layer_nodes)
    else:
        raise ValueError(
            "adaptive_neural requires hidden_layer_nodes to be a positive integer "
            "or a list of positive integers (possibly empty)."
        )
    if any(type(width) is not int or width <= 0 for width in hidden_widths):
        raise ValueError(
            "adaptive_neural hidden_layer_nodes must contain positive integers."
        )
    if type(output_dimension) is not int or output_dimension != 1:
        raise ValueError("adaptive_neural must have output_dimension equal to 1.")
    return (input_dimension, *hidden_widths, output_dimension)


def neural_parameter_count(architecture: tuple[int, ...]) -> int:
    """Count the weights and biases of every fully connected affine layer."""

    return sum(
        (source_width + 1) * destination_width
        for source_width, destination_width in pairwise(architecture)
    )


def validate_neural_input_dimension(
    input_dimension: int, observable_count: int
) -> None:
    """Check the input width against the detector's observable count."""

    if input_dimension != observable_count:
        raise ValueError(
            "adaptive_neural input_dimension must equal the number of detector observables."
        )
