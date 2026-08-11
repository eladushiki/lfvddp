# Project

## What This Is

We develop a statistical method to look at pairs of allegedly physically equivalent datasets of collider events, and estimate the probability with which they did not stem from the same probability distribution function.

## Core Value

Automatically scan pairs of lepton-flavor-differentiated HEP datasets for discrepancies from the Standard Model, without assuming the form or quality of a potential mismatch, and estimate the probability that it signals new physics.

## Project Shape

- **Complexity:** complex
- **Why:** The method must provide scientifically meaningful, model-agnostic new-physics evidence from HEP event data and be validated against injected-signal benchmarks.
- **Web stack:** not a web UI; Python workflow running locally and on the WIS ATLAS cluster through Singularity

## Users

- Experimentalists use observed discrepancies to direct theory development and may create or modify configurations.
- Developers maintain the code and may also create or modify configurations.

## Canonical Workflow

1. Choose dataset, detector, training, and plotting configurations.
2. Run or submit training for a lepton-flavor-differentiated dataset pair.
3. Aggregate test-statistic distributions from run outputs to calculate the reported significance `Z`, and produce diagnostic plots.

## Implemented Training Architecture

- LFVNN is the single implemented model family.
- One training job orchestrates each run, training paired numerator and denominator instances over the required `A_SR`, `A_CR`, `B_SR`, and `B_CR` dataset categories.
- Training behavior is configured through epoch, checkpoint, network-width, input-dimension, initialization, learning-rate, and runtime settings.
- The run persists model weights and test-statistic history for later aggregation.

## Statistical Training Invariant

The paper's symmetrized test statistic with efficiency weights is minimized in two parts: one over \(f(x)\) for dataset A and one over \(g(x)\) for dataset B. This mathematical objective, rather than the particular job orchestration, model implementation, or runtime mechanics, is the scientific invariant that future training changes must preserve.

## Constraints

- The probability estimate must distinguish potential new physics from statistical fluctuations and detector effects; detector effects are controlled before comparison.
- Dataset-pair matching is currently enforced through manually maintained paired configurations.
- Mock detector effects are examples only; the method and its theory must support arbitrary detector effects. Current effects are predefined stochastic functions that misdetect events and omit them from the final compared datasets.
- Code must remain mathematically equivalent to the paper's equations; implementation decisions not specified by the paper may evolve independently.
- The method must not assume a predefined new-physics mismatch hypothesis.

## Current State

The project develops and benchmarks a statistical method that compares pairs of lepton-flavor-differentiated collider-event datasets. Its primary benchmark is measured significance as a function of injected signal significance.

## Known Limitations

- The number of chi-squared degrees of freedom inferred while training the test-statistic `t` distribution is not yet explained; this does not currently prevent interpreting Z.
