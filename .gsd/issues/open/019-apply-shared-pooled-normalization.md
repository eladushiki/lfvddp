# Issue 019: Apply one shared pooled normalization to every dataset category

**Status:** open

## Problem

`DataBatch.get_normalized` derives one `ShiftAndNormalizationFactor` from the
pooled batch, but applying that factor through `DataSet.__truediv__` recomputes
the minimum of each target dataset. Consequently, A-SR, B-SR, A-CR, and B-CR
are not transformed by the same affine coordinate map.

For a pooled observable with minimum `m` and scale `s`, every event must be
mapped using the same transformation, equivalent to

```text
x_normalized = (x - m) / s - 1
```

The current implementation additionally depends on the category-specific
minimum. Categories with different sample minima therefore receive different
coordinate shifts even when they were generated from the same physical
distribution. This creates label-dependent preprocessing under the null and
can bias the learned test statistic. Adaptive sigmoid slopes may amplify even a
small category-dependent displacement.

The existing normalization tests call `get_normalized` separately on each
`DataSet`, so they do not exercise the pooled `DataBatch` path and do not catch
this defect.

## Required solution

- Give `ShiftAndNormalizationFactor` one explicit affine-transform meaning per
  observable. Its stored offset and factor must be the single source of truth.
- Make `DataSet.__truediv__` apply only the supplied transform. It must not
  derive an offset, minimum, maximum, or span from the target dataset.
- Make `DataSet.__mul__` the exact inverse of that same supplied transform,
  including when the factor was derived from a different dataset or a pooled
  batch.
- Preserve the existing finite behavior for constant observables.
- Keep `DataBatch.get_normalized` responsible for deriving the transform once
  from `unified_data` and applying that identical transform to every category.
- Reuse one implementation of the forward and inverse affine operations; do
  not duplicate normalization arithmetic between batch and dataset code.

## Acceptance criteria

- A batch whose categories have deliberately different minima is normalized
  with one common coordinate map.
- Concatenating the normalized categories gives the same event coordinates as
  normalizing the concatenated pooled dataset once and slicing it back into
  the original categories.
- Applying a factor derived from one dataset to another dataset or subset does
  not depend on the target dataset's extrema.
- Multiplying any such normalized dataset by the same factor reconstructs its
  original values within floating-point tolerance.
- Constant observables remain finite and round-trip correctly.
- Tests cover the `DataBatch.get_normalized` path with A-SR, B-SR, A-CR, and
  B-CR categories whose minima differ.
- Existing dataset-normalization and training tests continue to pass.
- Any user-facing description of normalization is updated if its stated
  semantics differ from the corrected shared affine transformation.

## Scope notes

- Do not alter dataset generation, event labels, or the statistical loss as
  part of this issue.
- Do not require each category to span `[-1, 1]`; only the pooled batch should
  determine that range. Individual categories may occupy strict subranges.
- After implementing the fix, rerun a background-only null ensemble before
  comparing the observed test-statistic distribution with earlier plots,
  because the corrected preprocessing intentionally changes model inputs.
