# Issue 001: Bound 2-D prediction meshes to observed data

**Status:** Done

## Problem

For prediction-progress plots with at least two observables, the lower two 3-D prediction panels displayed model values across the entire rectangular x-y mesh. Values far outside the observed data region are extrapolations with no useful interpretation.

## Implementation instructions

1. Build a two-dimensional polygon from the origin `(0, 0)` and all existing data points projected onto the two displayed observables.
2. Use the convex hull of those points as the tight enclosing polygon.
3. Build a mesh-membership mask for the prediction coordinates. Keep points in or on the hull and mask all outside points as `NaN`.
4. Apply the same mask to every prediction contour used in both lower 3-D panels before the existing surface renderer calculates finite surface polygons.
5. Keep the top distribution panels and one-dimensional plot behavior unchanged.

## Acceptance criteria

- Both lower 3-D prediction panels omit predictions outside the origin-plus-data convex hull.
- Predictions on or inside the hull remain displayed.
- The mask works with arbitrary plotted coordinate ranges.
- A unit test covers points inside, on, and outside a representative hull.

## Delivered

Implemented in `plot/plot_utils.py` and applied in `plot/plots.py`; regression coverage lives in `test/test_plot_utils.py`.
