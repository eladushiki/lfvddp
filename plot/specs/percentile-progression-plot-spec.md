# Percentile Progression Plot Specification

## Status

- **Implementation status:** Up to date
- **Primary implementation:** `plot/plots.py`: `t_train_percentile_progression_plot`
- **Shared filtering utilities:** `plot/plot_utils.py`: `_t_distribution_included_mask`
- **Plot scope:** Single submission

## Purpose

Show how selected empirical test-statistic percentiles evolve over training and
compare them with the corresponding diagnostic chi-square quantiles. The plot is diagnostic:
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
  performed independently at each checkpoint.

## Rendering

Each sample has one vertically stacked panel sharing the epoch axis. Every panel
contains empirical 2.5%, 25%, 50%, 75%, and 97.5% percentile curves. For
all positive diagnostic degree counts, dashed horizontal lines and their legend
entry are always shown. The result aggregator reconstructs the count from saved
run contexts using the existing statistical-calibration function, shared with the t-distribution plot:
adaptive LFVDDP counts all signal-network weights and biases at any depth, NPLM uses
the raw signal-network parameter count, and fixed families use their constrained
dimension. Zero-dimensional spaces raise an explicit error. This diagnostic
reference excludes shared nuisance parameters and is independent of multi-run
significance calibration. Parameter counting alone does not establish Wilks validity.
Neural input width comes from saved detector observables, and output width is
always one; only the hidden widths are configurable.

The horizontal axis is the configured training epoch and uses scientific
notation when appropriate. The vertical axis starts at zero. Its upper limit is
the largest non-negative empirical percentile in the final half of the training
history, or theoretical percentile shown in that panel, plus 5% headroom; it
never defaults below one. This may clip transient values from the left half so
that the convergence region remains visible. Negative intermediate percentiles
are clipped by the documented non-negative display range.

## Output Contract

- **Return type:** Matplotlib `Figure`.
- **Saving:** The plot factory/calling workflow persists the figure.
- **Configured output name:** `t_train_percentile_progression_plot`, subject to
  the factory's normal output naming.

## Acceptance Criteria

- [x] Percentiles use the shared final-statistic quality selection.
- [x] Complete histories are retained for selected runs.
- [x] Chi-square quantile references use the same configured diagnostic count as the t-distribution plot.
- [x] Adaptive neural and NPLM modes retain all five reference lines and the reference legend entry.
- [x] Each y-axis covers zero through the final half of non-negative curves and
  all non-negative reference curves, with 5% headroom.
- [x] The figure is reproducible from recorded submission results and config.
