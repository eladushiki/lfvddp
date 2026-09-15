# Issue 018: Add configurable signal-hypothesis function spaces

**Status:** open

## Description

Make the signal-hypothesis function space selectable through configuration,
using the same kind of mode choice that already exists for nuisance fitting.
To be clear, that is the parametrized implementation of the f function.

The adaptive neural network must remain the backward-compatible default.
Implementations should be shared between nuisance and signal (f), and either
should be able to choose from the implemented options.
Implementation should be by the open-closed principal, expecting future
implementations of more adaptive functions while their infrastructure remains.
An adequate variant of each mode should be derived from the current run dimension.

Support these top-level modes:

1. `adaptive_neural`: the current signal implementation, bitwise equivalent to
   existing behavior when no new option is configured.
2. `nplm`: the existing legacy "train like NPLM" behavior, preserved as a selectable
   special case.
3. `bin_indicators`: one trainable coefficient per configured detector bin,
  sharing bin setup and lookup machinery with the existing binwise nuisance
  implementation where appropriate.
4. `fixed_sigmoid`: Existing nn sigmoid features with configured, non-trainable
   centers and widths; centers spread evenly across the range, configurable
   widths with some reasonable range dependent default. Such that only their
   linear output coefficients are fitted. Configurable number of neurons.
4. `cubic_bspline`: cubic B-spline basis functions with fixed knots.
5. `orthogonal_polynomial`: Legendre or Chebyshev functions through a configured
  maximum degree on the normalized observable domain.

Basis geometry must not be selected from the A/B labels being tested. It may be
specified directly, derived from physics or simulation, or learned from an
independent reference sample. The adaptive-neural mode remains available when
data-driven feature location and scale are required; its null distribution is
to be calibrated empirically rather than inferred from raw parameter count.

## Mehtod
- Split to several milestones - the infrastructure overhaul itself, then each
  adaptive model extension and its tests.
- Separate pr for each part, tested on its own.

## Feature implementation instructions

- **RESEARCH:** map the current signal, NPLM, nuisance, configuration,
  checkpoint, prediction, and plotting responsibilities.
- **PLAN:** define a class hierarchy with one signal-function-space interface
  and no duplicated loss or evaluation logic. Treat bin indicators as one
  fixed-basis implementation and reuse equivalent binwise nuisance machinery.
- **CONFIGURE:** use one mode parameter plus one mode-specific dictionary.
  Validate required and forbidden keys for every mode through the normal
  configuration path.
- **IMPLEMENT:** preserve the existing adaptive-neural numerical path as the
  default and add the new implementations incrementally.
- **TEST:** run all relevant unit, integration, and system tests. Add focused
  coverage for basis construction, evaluation, rank, serialization, and
  end-to-end training or prediction.
- **DOCUMENT:** record the statistical distinction between adaptive feature
  search and fixed identifiable bases, including the need for empirical null
  calibration in adaptive mode.
- **UPDATE:** migrate tracked cluster config packs to the new configuration
  format without changing their behavior, and add background-only comparison
  packs for the new modes.

## Degrees-of-freedom requirements

- In any case, the degree-of-freedom count should be the just the number of
  de facto trainable parameters.
- Then, explicitly represent the observed-count constraint that removes the constant
  direction. For 13 independent basis functions including the constant, the
  expected regular rank is generically 12.
- Do not infer the fixed-basis degrees of freedom from a generic neural raw
  parameter count.
- For a fixed basis, compute or validate the rank of the evaluated design
  matrix after applying the statistic's normalization constraints.
- Surface the resulting rank in plot metadata and use it for analytic
  chi-square overlays only in regular fixed-basis modes.
- Adaptive-neural and NPLM modes must not claim a Wilks degree count solely from
  the number of trainable weights.

## Acceptance criteria

- The function-space mode is configurable through the existing configuration
  composition and validation path.
- Existing configurations select `adaptive_neural` implicitly and retain their
  intended numerical behavior.
- The current NPLM behavior is available through the explicit `nplm` mode. Rest
  plug into our proprietary DifferentiatingModel in the setup and loss calculation.
- Loss function remains the same and is valid for any f-model.
- All fixed-basis families can be constructed and evaluated in supported
  observable dimensions with correct dtype and device behavior.
- All modes share common setup/evaluation code with equivalent nuisance
  functionality.
- Cubic B-spline bases form the expected local overlapping partition over the
  configured domain and handle boundary knots correctly.
- Fixed-sigmoid and future bases honor its configured order,
  centers, and widths without making those geometry values trainable.
- Signal evaluation, loss calculation, training, checkpoint save/restore,
  prediction, and plotting work for every supported mode.
- Invalid modes, missing parameters, incompatible keys, invalid knot sequences,
  nonpositive widths, and rank-deficient fixed-basis configurations fail with
  useful validation errors.
- Tests verify linearity in the fixed-basis coefficients (except for the adaptive
  neural mode) and show that subtracting two coefficient vectors produces the
  same function as subtracting their evaluated functions.
- User-facing configuration documentation and examples are updated.

## Scope notes

- Do not reduce the flexibility or change the defaults of the adaptive neural
  discovery model.
- Choosing among several fixed bases after examining the same A/B labels is
  itself a trials factor. Either predeclare the basis or include that selection
  in the empirical null calibration.
- Add background-only cluster configuration packs for at least bin indicators,
  cubic B-splines, and fixed sigmoids.
