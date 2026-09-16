"""Shared option values for function-space unit tests."""

FUNCTION_SPACE_OPTIONS = {
    "adaptive_neural": {"input_dimension": 1, "hidden_layer_nodes": 2},
    "bin_indicators": {"minima": [0.0], "maxima": [3.0], "number_of_bins": [3]},
    "cubic_bspline": {"knots": [0.0, 1.0, 2.0, 3.0]},
    "orthogonal_polynomial": {
        "basis": "legendre",
        "maximum_degree": 2,
        "domain": [0.0, 3.0],
    },
    "fixed_sigmoid": {"centers": [-1.0, 1.0], "widths": [0.5, 0.5]},
    "gaussian_radial_basis": {"centers": [-1.0, 1.0], "widths": [0.5, 0.5]},
}
