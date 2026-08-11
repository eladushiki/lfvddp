# Issue 002: Split plot creation by execution scope

**Status:** Done

## Problem

One plotting entry point served two different workflows: overview plots for one submitted training batch and aggregate plots spanning many runs. The performance plot required callers to manually supply separate background and signal directory trees.

## Implementation instructions

1. Preserve `plot/create_plots.py` as the single-submission command-line interface and route it only to overview plots.
2. Add a multi-run entry point that accepts a root directory of recursive run outputs plus plotting configuration paths.
3. Define plot-to-entry-point membership in `PlotFactory` as a hardcoded, single source of truth:
   - single-submission: distribution, progression, data, and prediction-process plots;
   - multi-run: `performance_plot`.
4. For the multi-run entry point, recursively load run contexts and locate the outermost directory for which every descendant context is background-only.
5. Inject that directory as the background source. Use the complete supplied root as the signal search root, excluding contexts below the discovered background directory.
6. Retain all existing plotting configuration, figure persistence, and dimensional plot-name inference behavior.
7. Document both commands and include `performance_plot` in the basic plotting configuration without letting the single-submission entry point invoke it.

## Acceptance criteria

- `create_plots.py` creates only single-submission plots.
- `create_performance_plots.py` creates only multi-run plots.
- The performance entry point finds the outermost all-background subtree automatically.
- Background contexts are not reclassified as signal contexts.
- Plot selection and discovered directories are covered by unit tests.

## Delivered

Implemented in `plot/create_plots.py`, `plot/create_performance_plots.py`, `plot/plot_factory.py`, `plot/plot_utils.py`, and `plot/plots.py`; documented in `README.md`.
