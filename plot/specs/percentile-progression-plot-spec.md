# Percentile Progression Plot Specification

## Status

- **Implementation status:** Up to date
- **Primary implementation:** `plot/plots.py`: `t_train_percentile_progression_plot`
- **Shared filtering utilities:** `plot/plot_utils.py`: `_t_distribution_included_mask`
- **Plot scope:** Single submission

## Purpose

Show how selected empirical test-statistic percentiles evolve over training and
compare them with the corresponding theoretical chi-square quantiles. The plot
is diagnostic: it should reveal convergence behavior without allowing failed
training runs to flatten the valid curves.

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
contains empirical 2.5%, 25%, 50%, 75%, and 97.5% percentile curves and dashed
horizontal lines for the matching chi-square quantiles.

The horizontal axis is the configured training epoch and uses scientific
notation when appropriate. The vertical axis starts at zero. Its upper limit is
the largest non-negative empirical or theoretical percentile shown in that
panel, plus 5% headroom; it never defaults below one. Negative intermediate
percentiles are clipped by the documented non-negative display range.

## Output Contract

- **Return type:** Matplotlib `Figure`.
- **Saving:** The plot factory/calling workflow persists the figure.
- **Configured output name:** `t_train_percentile_progression_plot`, subject to
  the factory's normal output naming.

## Acceptance Criteria

- [x] Percentiles use the shared final-statistic quality selection.
- [x] Complete histories are retained for selected runs.
- [x] Empirical and matching chi-square quantiles are distinguishable.
- [x] Each y-axis covers zero through all non-negative curves with 5% headroom.
- [x] The figure is reproducible from recorded submission results and config.
