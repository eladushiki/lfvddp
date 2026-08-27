# Issue 012: Filter convergence plots

**Status:** Open

## Statement

Improve the t-distribution and percentile-progression plots by excluding invalid or non-converged training runs from the relevant reference populations.

## Acceptance criteria

- The lower t-distribution reference boundary is never below zero.
- Percentile progression uses only runs whose final recorded t history is finite, while retaining each selected run's complete history.
- Existing plotting behavior remains unchanged for already-valid histories.
