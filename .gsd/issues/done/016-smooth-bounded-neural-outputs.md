# Issue 016: Smooth bounded neural outputs

**Status:** Done

## Statement

Replace the hard neural output clamps for the reciprocal signal shift `f` and
neural nuisance `theta` with one shared smooth bounded parameterization. Keep
the existing log-safe open interval while allowing gradients to guide an output
back from values that would previously have been trapped outside the clamp.

## Acceptance criteria

- Neural `f` and `theta` use the same bounded-output implementation during
  training and prediction.
- Both outputs remain strictly inside the existing `(-1, 1)` safety margin.
- Raw outputs beyond the former clamp boundary retain a non-zero gradient.
- Scalar-binned nuisance behavior is unchanged.
- The convergence-oriented parameterization is documented.

## Completion note

Added one shared scaled-`tanh` likelihood-shift parameterization and routed the
signal and neural nuisance estimators through it for both training and
prediction. Boundary regression tests force raw outputs past the old clamp and
verify strict bounds together with non-zero recovery gradients.
