---
name: create-configurations
description: Create or update LFVDDP dataset and plot configurations, including signal event counts calibrated to a target continuous injected significance.
---

# Create configurations

Use this skill when creating or changing LFVDDP configuration files for generated
datasets, training, or significance plots.

- Treat the configuration's generator specifications, dimension, background
  event count, integration domain, and target significance as the source of
  truth. Do not infer target values from names such as `significance-01`.
- For signal event counts, use the repository utility
  `calibrate_signal_events` (or
  `data_tools.signal_calibration.calc_n_signal_events_for_generated_signal`).
  It solves the same continuous injected-significance calculation used by the
  plotter, using the full signal and background PDFs over the configured domain.
- Pass the actual background count for the region and the complete signal
  specification, including location and width. Do not estimate the yield from
  a local PDF maximum or a fixed cell width.
- Generate related configurations from explicit shared parameters so 2D/4D,
  background-size, and signal-width variants cannot silently diverge. Keep
  provisional calibration values clearly identified when the background is an
  approximation of a loaded sample.
- After editing, recalculate the forward significance for every target and run
  the focused configuration/calibration tests. Do not rewrite unrelated local
  or archived configuration files.

Example:

```bash
calibrate_signal_events \
  --background-generator '{"function":"exponential_background"}' \
  --signal-generator '{"function":"gaussian_signal","arguments":{"location":4.0,"gaussian_signal_sigma":0.16}}' \
  --number-of-dimensions 2 \
  --background-events 25000 \
  --target-significance 1 2 3 4 5
```
