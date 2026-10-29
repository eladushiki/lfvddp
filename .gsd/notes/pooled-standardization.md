# Pooled mean / standard-deviation normalization

User request: center and scale inputs for new fits without adding another
training restart. New LFVDDP fits use each observable's mean and population
standard deviation across all four detected categories, weighted by event count.
One shared map is fitted once, then reused for training, prediction, and physical
function-space geometry. Per-category centering would erase physical differences
and is deliberately excluded. This changes initialization and optimizer paths;
it does not establish better convergence for every dataset.

Exact constant observables use their observed value and unit scale. Detect
constants before computing moments: floating-point averaging of repeated large
(or non-binary decimal) constants can otherwise invent a nonzero variance.

## Compatibility decision

The affine factor now serializes an explicit output `shift`. New standardization
uses zero; a missing shift retains the historical minus-one translation. This
keeps historical constructors, checkpoint metadata, and directly pickled trained
artifacts meaningful. Retaining actual means avoids encoding `mean - std` in the
offset, which can lose the intended shift for large constants. Continued training
always reuses the saved map rather than recomputing new moments. The existing
checkpoint structural fingerprint excludes normalization values and is unchanged.

## Caller and entry-point inventory

- `DataSet.get_normalized` derives the map; division/multiplication use the one
  shared forward/inverse implementation. Dataset tests cover multiple observables,
  constants, pooled categories with distinct minima, subsets, inverse maps, new
  serialization, historical mappings, and historical pickled dataclasses.
- `DataBatch.get_normalized` derives statistics from `unified_data` once and
  applies the same map to A-SR, B-SR, A-CR, and B-CR. Dataset tests compare this
  directly with pooled normalization and check its population moments.
- `DifferentiatingModel._prepare_training_data` is the production caller.
  Existing training, function-space integration, baseline, role, and checkpoint
  tests exercise null/alternative, disabled/binned/neural nuisance, 1D/2D,
  shallow/deep/no-hidden neural networks, and all deterministic families.
- Primary, secondary, and nuisance prediction share the stored affine map.
  Existing checkpoint and integration matrices exercise all three public APIs.
- Checkpoint continuation restores the factor before geometry construction.
  Continuation tests now cover both new maps and historical metadata without
  `shift`, across enabled neural nuisances, neural depths, and both fit roles.
- Bin edges, spline knots, polynomial domains, fixed-sigmoid centres/widths,
  and Gaussian-RBF centres/widths call `normalize_values` / `scale_values`.
  Physical-geometry tests and function-space integration cover these paths.
- Worker result restoration transfers the affine dataclass directly. The new
  field is retained by ordinary pickling; a legacy-pickle regression confirms
  missing instance fields retain historical semantics.
- Prediction plotting uses the model public APIs. Existing 1D/2D prediction-grid
  and plot-generation integration checks exercise these maps. No plotting code
  or dimensional plot specification changes are required.
- NPLM owns its preprocessing and does not call this normalization path.
  Its TensorFlow-dependent training test is unavailable in this macOS environment.

Issue 019's category-specific-offset bug is already corrected on the base branch;
its stale issue state is outside this feature's scope.

## Verification result

141 relevant tests passed (dataset, geometry, checkpoint, baseline, family
integration, role, training, and prediction plotting suites). Four exclusions:
two long convergence benchmarks, one TensorFlow-dependent NPLM training test,
and the existing 2D plot-generation assertion expecting four axes after the
base branch changed the figure to thirty axes. The same plotting assertion
fails with the base branch's original normalization method; figure generation
itself succeeds. The 2D prediction-grid comparison and 1D plot-generation test
pass. No production plotting code was changed.

Python files were formatted with Ruff. Ruff F/I checks pass; existing long lines and
data-utils type annotations retain their pre-existing lint warnings.
