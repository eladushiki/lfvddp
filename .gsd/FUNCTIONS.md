# Future Function Descriptions

## Prediction progress plot: restrict 3-D prediction mesh

**Status:** Implemented

When the prediction progress plot has two or more dimensions, restrict the prediction surface in its bottom two 3-D graphs to an x-y mesh inside a tight polygon that contains the origin `(0, 0)` and every existing data point.

Do not display predictions outside this polygon: the model is trained only on the existing points, so predictions beyond their enclosing region have no meaningful interpretation.

## Split plot-creation entry points by scope

**Status:** Implemented

Split the current plot-creation entry point into two entry points, selected by the type of plots being created:

- **Single-submission plots:** retain an entry point identical to the current `create_plots.py`, for plots that overview the results of one submission.
- **Multi-run plots:** add a separate entry point for plots that aggregate multiple runs, initially the performance plot.

The multi-run entry point must automatically locate the outermost directory whose recursive contents are exclusively background runs (runs with no signal events). Treat every other context as a signal context.

The plot factory must decide which plots each entry point invokes, using a hardcoded categorization of plot types.
