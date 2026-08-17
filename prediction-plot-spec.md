# Prediction Plot Specification

## Status

- **Document status:** Draft template
- **Implementation status:** Current behavior documented below
- **Primary implementation:** `plot/plots.py`: `plot_prediction_process_1d` and `plot_prediction_process_2d`
- **Primary utilities:** `plot/plot_utils.py`
- **Plot scope:** Single submission

## Purpose

Visualize the learned LFVDDP predictions alongside the observed signal-region (SR) and control-region (CR) distributions. The plot is intended to make the model's signal and null hypotheses, including nuisance terms, comparable across the selected observables.

> **To define:** What scientific question must a reader answer from this plot?

## Invocation and Inputs

### Required runtime inputs

| Input | Current behavior |
| --- | --- |
| Execution context | Supplies the merged dataset, detector, training, and plotting configuration. |
| Numerator training | Supplies the primary model, data batch, and numerator detector effect. |
| Denominator training | Supplies the denominator model and detector effect. |
| Observables | Uses the supplied `along_observables`; otherwise uses the configured detector observables. Only one observable is allowed for the 1D plot and two for the 2D plot. |

### Required dataset categories

| Region | Dataset categories | Derived background |
| --- | --- | --- |
| SR | `A_SR`, `B_SR` | `A_SR + B_SR` |
| CR | `A_CR`, `B_CR` | `A_CR + B_CR` |

> **To define:** Dataset selection, event filtering, weighting, and any required preprocessing assumptions.

## Current Figure Layout

### 1D prediction plot

A 2 × 2 figure with four panels:

| Position | Panel | Current contents |
| --- | --- | --- |
| Top left | SR distribution | Histograms of A-SR, B-SR, their combined background, and weighted numerator-model predictions. |
| Top right | CR distribution | Histograms of A-CR, B-CR, their combined background, and weighted denominator-model predictions. |
| Bottom left | SR prediction | Curves for numerator signal-hypothesis components and numerator nuisance factors. |
| Bottom right | CR prediction | Curves for denominator/null-hypothesis components and denominator nuisance factors. |

The distribution panels share synchronized output-axis limits. The figure has a configurable title; its current default is `Datasets Along the Process`.

### 2D prediction plot

The current code also provides an analogous 2D implementation, `plot_prediction_process_2d`, for exactly two selected observables. It uses the same SR/CR and numerator/denominator conceptual split, rendering compatible sliced 2D panels.

> **To define:** Whether 2D is part of this artifact's scope, and the expected reading order for the panels.

## Current Distribution Rendering

- Display-bin edges are derived from the unified data for the configured detector observables.
- The selected observable bins use `plot__prediction_process_number_of_bins`.
- The default configuration uses **30 bins**.
- The top panels draw A-region, B-region, and combined-background distributions.
- The model predictions are converted to weighted distributions using the corresponding combined background as the reference dataset.
- When `plot__prediction_process_normalize_each_prediction` is enabled, each distribution/prediction is normalized independently. The current default is **enabled**.
- Each distribution panel receives a prediction-process legend.

> **To define:** Required normalization convention, expected y-axis units, treatment of empty bins, error bars/uncertainties, and whether the raw count scale must also be available.

## Current Prediction Rendering

### SR panel: numerator model

The numerator model supplies:

| Quantity | Current label/meaning |
| --- | --- |
| `predict` | \(e^{f(x)}(1+\eta(x))\) signal-hypothesis component |
| `predict_secondary` | \(e^{g(x)}(1-\eta(x))\) signal-hypothesis component |
| `predict_eta` | Numerator nuisance factor \(\eta(x)\) |

The implementation also derives and displays the eta-removed signal terms \(e^{f(x)}\) and \(e^{g(x)}\), together with \(1+\eta(x)\) and \(1-\eta(x)\).

### CR panel: denominator model

The denominator model supplies `predict_eta`, which is displayed as the null-hypothesis nuisance factors \(1+\eta(x)\) and \(1-\eta(x)\).

### Common behavior

- A horizontal reference line is drawn at prediction value **1.0** in 1D prediction panels.
- Prediction values are evaluated over a spanning dataset built from detector-bin coordinates.
- The model output is projected onto the selected observable(s) before rendering.
- Product terms use dash-dot lines; component terms use solid lines; denominator/null terms use dashed lines.

> **To define:** Which curves are mandatory, whether derived curves should be visible by default, and the acceptable prediction range or clipping policy.

## Current Visual Encoding

| Element | Current encoding |
| --- | --- |
| Combined background | Gray |
| \(f\)-family / A component | Blue (`tab:blue`) |
| \(g\)-family / B component | Orange (`tab:orange`) |
| Lighter \(f\)-family variants | Cornflower blue / lightsky blue |
| Lighter \(g\)-family variants | Sandybrown / moccasin |
| Reference prediction | Gray dotted horizontal line at 1.0 |
| SR model-prediction markers | Circles for \(e^{f(x)}(1+\eta(x))\) |
| SR secondary-prediction markers | Squares for \(e^{g(x)}(1-\eta(x))\) |

The figure uses the global plotting configuration. The baseline configuration sets a white figure face, classic Matplotlib style, serif font family, font size 24, and figure size 10 × 9 inches.

> **To define:** Publication palette, accessibility requirements, typography, legend placement, axis-label conventions, and export resolution/formats.

## Configuration Contract

| Key | Current default | Effect |
| --- | ---: | --- |
| `plot__prediction_process_number_of_bins` | `30` | Number of display bins for the prediction-process distributions. |
| `plot__prediction_process_normalize_each_prediction` | `true` | Normalizes each displayed prediction/distribution independently. |
| `plot__figure_size` | `[10, 9]` | Base figure dimensions in inches. |

> **To define:** Per-plot overrides, validation limits, and the stable public configuration name for this plot.

## Output Contract

- **Return type:** Matplotlib `Figure`
- **Saving:** The plot function does not save the figure itself; the plot factory/calling workflow owns output persistence.
- **Current output name:** _To define_
- **Supported formats:** _To define_
- **Destination directory:** _To define_

## Acceptance Criteria

- [ ] The selected-observable dimensionality is validated before plotting.
- [ ] SR and CR distribution panels show the intended data and weighted predictions.
- [ ] Prediction panels show the intended numerator and denominator curves with unambiguous legends.
- [ ] Axis labels, units, and normalization are scientifically correct.
- [ ] Colors, line styles, and markers remain distinguishable in grayscale and for color-vision deficiencies.
- [ ] The artifact is reproducible from a recorded configuration and training output.
- [ ] A representative 1D and, if in scope, 2D output has been reviewed.

## Open Decisions

1. _What is the canonical plot title and caption?_
2. _Which curves should be shown by default versus optionally?_
3. _What uncertainties or confidence intervals must be displayed?_
4. _Should distributions be independently normalized, count-normalized, or both?_
5. _What output filename and publication-ready export settings are required?_
