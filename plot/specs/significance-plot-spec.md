# Significance Plot Specification

## Status

- **Implementation status:** Ongoing - up to date with code
- **Primary implementation:** `plot/plots.py`: `performance_plot`
- **Primary utilities:** `plot/plot_utils.py`: context discovery, grouping, and performance-curve calculation
- **Plot scope:** Multi-run

## Purpose

Show the measured LFVDDP significance across compatible signal runs relative to their ideal (analytic, if known) significance. The plot uses background-only runs as a common reference distribution, as well as the theoretical $\chi^2$ limit, so readers can compare observed sensitivity among signal models and data-generation settings.

A reader should be able to determine:

- How measured significance changes with ideal $Z=\sqrt{q_0}$.
- Which compatible signal-run groups each curve represents.
- The uncertainty on measured significance at each sampled ideal significance.

## Invocation and Inputs

| Input | Current behavior |
| --- | --- |
| Execution context | Supplies the plotting configuration and output lifecycle. Also, the degree of the expected $\chi^2$ distribution by the configured number of the models' degrees of freedom. |
| Background-only parent directory | Required; each outermost directory containing a context beneath it contributes to the reference `t` distribution. |
| Signal parent directory | Required; each outermost directory containing a context beneath it supplies one signal distribution. |

The plot factory discovers and injects the two parent directories for this multi-run plot. Signal contexts are grouped only when their dataset configurations are compatible.

## Aggregation and Calculation

- Background-only `t` values from all discovered background contexts are aggregated into one reference distribution.
- Signal contexts are discovered recursively and grouped by compatible dataset configuration.
- Each signal context contributes a `t` distribution and an ideal significance derived from its configured injected signal.
- For each signal distribution, the measured significance is the common-background percentile of that distribution's mean `t`; uncertainty is reported from mean `t` plus or minus one standard deviation.
- Invalid or incompatible context data is surfaced by the discovery and aggregation utilities rather than silently combined.

## Rendering

The figure contains one axes:

| Element | Rendering |
| --- | --- |
| Signal group | One labelled curve per compatible signal group. |
| Measured significance | Marker-and-line points against ideal significance. |
| Uncertainty | Error bars on measured significance. |
| Reference relation | The ideal-significance diagonal used to compare measured and ideal sensitivity. |

The horizontal axis is ideal significance $\sqrt{q_0}$; the vertical axis is measured significance. Labels are constructed from the group dataset configuration so the compared signal settings remain identifiable.

### Further requirements
- Every Carpenter figure reserves the same 12% bottom row for hash stamping, so that row can be cropped without hiding plot content.
- Convert the snake case signal names in legend to english with parameters in latex equations if needed.

## Configuration Contract

| Key | Current default | Effect |
| --- | ---: | --- |
| `plot__target_run_parent_directory` | `""` | Parent location from which multi-run plotting input is discovered. |
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
- [ ] Measured significance and its uncertainty are plotted against ideal significance.
- [ ] The figure is reproducible from the recorded parent-directory runs and configurations.
