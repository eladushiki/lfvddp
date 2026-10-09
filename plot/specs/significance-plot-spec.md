# Significance Plot Specification

## Status

- **Implementation status:** Ongoing - up to date with code
- **Primary implementation:** `plot/plots.py`: `performance_plot`
- **Primary utilities:** `plot/plot_utils.py`: context discovery, grouping, and performance-curve calculation
- **Plot scope:** Multi-run

## Purpose

Show the measured LFVDDP significance across compatible signal runs relative to their configured signal strength. For generated datasets, the signal-strength axis is the ideal analytic significance. For loaded datasets, the analytic background density is not known, so the signal-strength axis is the binned evident injected significance estimated from saved samples. The plot uses background-only runs as a common reference distribution so readers can compare observed sensitivity among signal models and data-generation settings.

A reader should be able to determine:

- How measured significance changes with generated-data ideal or loaded-data evident significance $Z=\sqrt{q_0}$, where $Z$ is injected significance and $q_0$ is the expected null-versus-signal likelihood-ratio statistic.
- Which compatible signal-run groups each curve represents.
- The uncertainty on measured significance at each sampled ideal significance.

## Invocation and Inputs

### Empty empirical background bins

For loaded datasets, exclude every zero-background bin from both terms of the
binned injected-significance sum, including its negative signal contribution.
This policy is shared by 1D and multidimensional performance plots. If any
worker excludes positive signal, show a red warning on the figure and label
each affected point with the mean and maximum ignored expected signal events
per worker. Counts are histogram weights scaled to the configured mean signal
yield, not raw sampled events or a sum across repeated experiments. Means and
maxima include all workers contributing to that point, including zero exclusions.
An all-excluded signal has zero restricted-bin significance, not infinite
significance. No pseudocount or replacement background model is introduced.
The displayed injected significance is restricted to the retained bins; measured
significance remains the result of the original training experiment.

| Input | Current behavior |
| --- | --- |
| Execution context | Supplies the plotting configuration and output lifecycle. Also, the degree of the expected $\chi^2$ distribution by the configured number of the models' degrees of freedom. |
| Background-only parent directory | Required; each outermost directory containing a context beneath it contributes to the reference `t` distribution. |
| Signal parent directory | Required; each outermost directory containing a context beneath it supplies one signal distribution. |

The plot factory discovers and injects the two parent directories for this multi-run plot. Signal contexts are grouped only when their dataset configurations are compatible.

## Aggregation and Calculation

- Background-only `t` values from all discovered background contexts are aggregated into one reference distribution.
- Signal contexts are discovered recursively and grouped by compatible dataset configuration.
- Each signal context contributes a `t` distribution and an x-axis signal-strength value:
  - generated datasets use detector-level analytic injected significance derived from their existing generated background and signal PDFs. Multiply both event densities by the configured nominal efficiency for the signal dataset’s A/B family, respecting detector observable selection and order. Integrate the full local likelihood contribution, including the accepted signal subtraction, without renormalizing accepted yields. Efficiency uncertainty, measurement smearing, and nuisance profiling are outside this efficiency-only benchmark;
  - loaded datasets use an evident injected significance calculated from expected background and signal counts in detector-observable bins, because they do not define analytic background PDFs. Training saves the sampled background and signal data plus the prediction plot's exact bin edges in `data_samples.npz`; plotting reads this artifact from each successful training worker, using that worker's saved configuration and bin edges. Array submission contexts remain the grouping boundary and are not sample owners. The horizontal coordinate and error bar are the mean and standard deviation of the worker-level significances, including workers from repeated submissions at the same configured signal strength. Standalone runs continue to use their own artifact. A missing artifact in a successful worker is an error identifying its path rather than a reason to draw a new random sample or silently drop the worker. This lookup is shared by 1D and n-dimensional loaded-data plots.
- Mixed generated and loaded datasets are rejected in one performance plot. They use different x-axis semantics and must not be overlaid.
- A loaded-data worker with zero sampled injected signal events contributes zero injected significance, even when its configured mean signal count is positive. Keep that worker in the aggregate; its ignored-signal-event count is also zero. Missing sample artifacts remain errors.
- For each signal distribution, the measured significance is the common-background percentile of that distribution's median `t`; uncertainty is reported from median `t` plus or minus one standard deviation.
- Invalid or incompatible context data is surfaced by the discovery and aggregation utilities rather than silently combined. If outlier filtering leaves no usable background values, the error identifies the checked directories and raw, finite, and finite-nonnegative counts. A verified `array-job-artifacts.tar.gz` created before aggregate plotting must be restored rather than misclassified as a failed residue.

## Rendering

The figure contains one axes:

| Element | Rendering |
| --- | --- |
| Signal group | One labelled curve per compatible signal group. |
| Measured significance | Marker-and-line points against the plot's signal-strength axis. |
| Gaussian-fit significance | Dashed curve through the Gaussian-fit estimate at each sampled signal strength. |
| Uncertainty | Error bars on measured significance. |
| Reference relation | The ideal-significance diagonal is drawn only for generated datasets, where the horizontal axis is analytic significance. |

The horizontal axis is detector-level ideal significance $\sqrt{q_0}$ for generated datasets and evident injected significance $\sqrt{q_0}$ for loaded datasets; the vertical axis is measured significance. For loaded datasets, the evident injected significance uses the exact prediction plot display bins computed from the detected training batch and controlled by `plot__prediction_process_number_of_bins`. The bin counts are scaled to the configured mean background and signal event counts before evaluating the binned likelihood-ratio formula. Generated datasets connect measured-significance points and draw a band from the measured-significance spread. Loaded datasets keep measured-significance points unconnected while their Gaussian-fit estimates remain connected by a dashed curve. Labels are constructed from the group dataset configuration so the compared signal settings remain identifiable.

### Further requirements
- Every Carpenter figure reserves the same 12% bottom row for hash stamping, so that row can be cropped without hiding plot content.
- This one-panel plot uses Carpenter's standard left, right, top, and bottom borders, shared with the percentile-progression and t-distribution plots.
- Convert the snake case signal names in legend to english with parameters in latex equations if needed.

## Configuration Contract

| Key | Current default | Effect |
| --- | ---: | --- |
The multi-run plotting input is the first positional command-line argument, stored as `config__out_dir`.
| `plot__figure_size` | `[10, 9]` | Figure dimensions in inches. |
| `plot__pyplot_styling` | Basic plot config | Global Matplotlib typography and style. |
| `plot__figure_styling` | Basic plot config | Figure appearance settings. |

The plot has no per-curve instruction parameters in the basic configuration; its inputs are discovered from the supplied run directories.

## Output Contract

- **Return type:** Matplotlib `Figure`.
- **Saving:** The plot factory/calling workflow persists the figure.
- **Configured output name:** The plot instruction's `name`, subject to the factory's normal output naming.

## Acceptance Criteria

- [ ] Background-only contexts are aggregated into one reference distribution.
- [ ] Signal contexts are grouped only with compatible dataset configurations.
- [ ] Every displayed curve has a readable configuration-derived label.
- [ ] Measured significance and its uncertainty are plotted against the correct signal-strength axis for the dataset source type.
- [ ] Loaded datasets use binned evident injected significance instead of analytic PDF integration.
- [ ] Mixed generated and loaded datasets are rejected before plotting.
- [ ] The figure is reproducible from the recorded parent-directory runs and configurations.

The generated-data efficiency adapter and likelihood integration are shared by 1D and n-dimensional plots and by saved-run significance aggregation. Aggregation caches include detector configuration so otherwise identical runs with different efficiencies cannot share a cached significance.
