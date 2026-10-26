# Percentile Progression Plot Specification

## Status

- **Implementation status:** Up to date
- **Primary implementation:** `plot/plots.py`: shared `_train_percentile_progression_plot` renderer with test-statistic, numerator, and denominator entry points
- **Shared filtering utilities:** `plot/plot_utils.py`: `_t_distribution_included_mask`
- **Plot scope:** Single submission

## Purpose

Show how selected empirical test-statistic, numerator-loss, and denominator-loss
percentiles evolve over training in three separate figures. Compare only the
test-statistic figure with diagnostic chi-square quantiles. The plot is diagnostic:
it should reveal convergence behavior without allowing failed training runs to
flatten the valid curves.

## Data Selection and Filtering

- Histories are grouped by sample and aligned on their recorded epochs.
- A run is selected from its final recorded `t` value using the same quality
  rule as the t-distribution plot.
- Non-finite and negative final values are invalid. Finite lower- and upper-tail
  outliers are identified relative to the central 5%-95% reference population
  and excluded as non-converged or overfitted respectively.
- Tail thresholds are four central-reference standard deviations from its mean.
- Once selected, a run contributes its complete history; selection is not
  performed independently at each checkpoint. The same final-`t` selection is
  applied to the numerator and denominator histories, preserving the same run
  population across all three figures. Component losses are not filtered by
  their own sign or magnitude.

## Rendering

Each sample has one vertically stacked panel sharing the epoch axis. Every panel
contains empirical 2.5%, 25%, 50%, 75%, and 97.5% percentile curves. For
all positive diagnostic degree counts, the test-statistic figure always shows
dashed horizontal reference lines and their legend entry. Numerator and
denominator figures show only empirical curves; they do not access chi-square
calibration or require saved run contexts for it. The result aggregator reconstructs the count from saved
run contexts using the existing statistical-calibration function, shared with the t-distribution plot:
adaptive LFVDDP counts all signal-network weights and biases at any depth, NPLM uses
the raw signal-network parameter count, and fixed families use their constrained
dimension. Zero-dimensional spaces raise an explicit error. This diagnostic
reference excludes shared nuisance parameters and is independent of multi-run
significance calibration. Parameter counting alone does not establish Wilks validity.
Neural input width comes from saved detector observables, and output width is
always one; only the hidden widths are configurable.

The horizontal axis is the configured training epoch and uses scientific
notation when appropriate. The test-statistic vertical axis starts at zero. Its upper limit is
the largest non-negative empirical percentile in the final half of the training
history, or theoretical percentile shown in that panel, plus 5% headroom; it
never defaults below one. This may clip transient values from the left half so
that the convergence region remains visible. Negative intermediate percentiles
are clipped by the documented non-negative display range.

Numerator and denominator axes instead fit both signed bounds to finite empirical
percentile values in the final half of training, with 5% of that range added at
each end. Early transients may be clipped. Constant curves receive padding of
5% of the larger of their absolute value and one; no finite values yields
`[-1, 1]`. Their y-axis labels name the component. Their debug titles identify
numerator or denominator; ordinary title suppression remains factory-controlled.
The same renderer serves every dataset dimensionality.

Run-hash stamps use Carpenter’s shared relative font height: 10 points per
9 inches of canvas height (about 1.54%), matching d_dist for every canvas size.

## Output Contract

- **Return type:** Matplotlib `Figure`.
- **Saving:** The plot factory/calling workflow persists the figure.
- **Configured output names:** `t_train_percentile_progression_plot`,
  `numerator_train_percentile_progression_plot`, and
  `denominator_train_percentile_progression_plot`, subject to the factory's
  normal output naming. Each entry returns one figure.
- Both standard generated and loaded plot configs enable all three figures.
  Custom configs opt into the new figures with separate entries and empty
  `instructions` objects; existing entries do not produce companion outputs.

## Acceptance Criteria

- [x] Percentiles use the shared final-statistic quality selection.
- [x] Complete histories are retained for selected runs.
- [x] Chi-square quantile references use the same configured diagnostic count as the t-distribution plot.
- [x] Adaptive neural and NPLM modes retain all five reference lines and the reference legend entry.
- [x] Each y-axis covers zero through the final half of non-negative curves and
  all non-negative reference curves, with 5% headroom.
- [x] The figure is reproducible from recorded submission results and config.

- [x] Component figures reuse final-`t` selection, percentiles, styling, and layout.
- [x] Component figures omit all chi-square references and calibration lookup.
- [x] Signed final-half limits display negative losses and constant denominators.
