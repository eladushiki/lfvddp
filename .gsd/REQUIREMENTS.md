# Requirements

This file is the explicit capability and coverage contract for the project.

## Active

### R001 — An experimentalist can compare a pair of lepton-flavor-differentiated HEP datasets and receive a model-agnostic probability estimate for new physics.
- Class: primary-user-loop
- Status: active
- Description: An experimentalist can compare a pair of lepton-flavor-differentiated HEP datasets and receive a model-agnostic probability estimate for new physics.
- Why it matters: This is the primary workflow for directing theory development from observed data discrepancies.
- Source: user
- Validation: Run the end-to-end comparison on a valid dataset pair and produce the probability estimate.
- Notes: The method must not require a predefined new-physics mismatch hypothesis.

### R002 — The method is benchmarked by the progression of measured significance as injected signal significance increases in otherwise Standard-Model simulated datasets.
- Class: quality-attribute
- Status: active
- Description: The method is benchmarked by the progression of measured significance as injected signal significance increases in otherwise Standard-Model simulated datasets.
- Why it matters: This measures the method's ability to detect discrepancies without assuming their form in advance.
- Source: user
- Validation: Run injected-signal studies and produce the measured-significance-versus-injected-significance progression.
- Notes: Higher measured significance at a given injected signal significance is better.

### R003 — The comparison workflow accepts any supported HEP dataset source when paired datasets differ by lepton flavor and are otherwise matched as closely as possible; it warns when the pair is not sufficiently matched.
- Class: core-capability
- Status: active
- Description: The comparison workflow accepts any supported HEP dataset source when paired datasets differ by lepton flavor and are otherwise matched as closely as possible; it warns when the pair is not sufficiently matched.
- Why it matters: Experimentalists must be able to apply the method to the datasets relevant to their analysis rather than only a single source type.
- Source: user
- Validation: Verify that pair construction preserves matching selections, observable definitions, background composition, and detector corrections except for lepton flavor, and that detected discrepancies warn with the differing configuration fields and affected dataset categories.
- Notes: Physical comparability is the basis for interpreting a discrepancy as potential new physics rather than a dataset mismatch. Matching is currently maintained manually.

### R004 — Each comparison persists its complete configuration including random seed, machine-readable test-statistic distributions and results needed to calculate significance Z, the significance-versus-injected-significance performance plot, and trained model weights.
- Class: integration
- Status: active
- Description: Each comparison persists its complete configuration including random seed, machine-readable test-statistic distributions and results needed to calculate significance Z, the significance-versus-injected-significance performance plot, and trained model weights.
- Why it matters: Experimentalists need plots to assess findings and structured outputs to incorporate them into downstream analysis.
- Source: user
- Validation: Run an end-to-end injected-signal comparison and verify the saved configuration and seed, test-statistic distributions, derived Z result, performance plot, and trained model weights.
- Notes: Configuration and seed reproduce the run. The test-statistic distribution enables later Z recalculation. Trained model weights are preserved but are secondary to the configuration/seed and statistical artifacts for reproducibility. The output remains model-agnostic with respect to the new-physics mismatch.

### R005 — The established workflow supports local execution and WIS ATLAS cluster execution through Singularity containers.
- Class: operability
- Status: active
- Description: The established workflow supports local execution and WIS ATLAS cluster execution through Singularity containers.
- Why it matters: Experimental studies must be runnable in development and scalable cluster environments.
- Source: user
- Validation: Run the documented workflow locally and submit the corresponding Singularity-backed cluster execution.
- Notes: Singularity runs on the WIS ATLAS cluster.

### R006 — Configuration validation runs before execution through x_validate and reports all independent invalid parameter values and contradictory parameter combinations with actionable errors.
- Class: failure-visibility
- Status: active
- Description: Configuration validation runs before execution through x_validate and reports all independent invalid parameter values and contradictory parameter combinations with actionable errors.
- Why it matters: Experimentalists and developers must detect invalid study setup before consuming local or cluster resources.
- Source: user
- Validation: Run a configuration containing multiple independent invalid values and contradictions; verify that validation fails before execution and reports every violated condition.
- Notes: Applies to configurations created by experimentalists and developers.

## Validated

## Deferred

## Out of Scope

## Traceability

| ID | Class | Status | Primary owner | Supporting | Proof |
|---|---|---|---|---|---|
| R001 | primary-user-loop | active | none | none | Run the end-to-end comparison on a valid dataset pair and produce the probability estimate. |
| R002 | quality-attribute | active | none | none | Run injected-signal studies and produce the measured-significance-versus-injected-significance progression. |
| R003 | core-capability | active | none | none | Verify that pair construction preserves matching selections, observable definitions, background composition, and detector corrections except for lepton flavor, and that detected discrepancies warn with the differing configuration fields and affected dataset categories. |
| R004 | integration | active | none | none | Run an end-to-end injected-signal comparison and verify the saved configuration and seed, test-statistic distributions, derived Z result, performance plot, and trained model weights. |
| R005 | operability | active | none | none | Run the documented workflow locally and submit the corresponding Singularity-backed cluster execution. |
| R006 | failure-visibility | active | none | none | Run a configuration containing multiple independent invalid values and contradictions; verify that validation fails before execution and reports every violated condition. |

## Coverage Summary

- Active requirements: 6
- Mapped to slices: 0
- Validated: 0
- Unmapped active requirements: 6
