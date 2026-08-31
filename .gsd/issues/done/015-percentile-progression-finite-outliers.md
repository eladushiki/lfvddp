# Issue 015: Filter finite percentile-progression failures

**Status:** Done

## Statement

Make percentile progression use the same final-statistic quality selection as
the t-distribution plot, including finite convergence failures, and give the
plot an explicit non-negative y-range derived from the curves it displays.

## Acceptance criteria

- Non-finite, negative, lower-tail, and upper-tail failed runs do not contribute
  to percentile progression.
- Selected runs retain their complete recorded histories.
- Tail thresholds use a central reference population that is not contaminated
  by either extreme.
- Every progression panel spans zero through the largest empirical or
  theoretical percentile curve, with only a small readability margin.
- The percentile-progression Markdown specification documents selection and
  axis behavior.

## Completion note

Both plots now use one shared final-statistic inclusion mask based on an
uncontaminated central reference population. Percentile panels explicitly span
zero through their largest empirical or theoretical curve with 5% headroom,
and regression tests cover the finite catastrophic tails observed in practice.
