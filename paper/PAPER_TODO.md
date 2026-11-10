# Paper TODO

Working checklist for Elad and collaborators. Items remain open until we review and complete them together.

## Scope and provenance

- Main document: [ddppaper2.tex](ddppaper2.tex), including its shared `common/sections/` sources.
- Reviewed local revision: `72d311e0`. Updating from remote `main` was blocked by Git metadata permissions and GitHub DNS resolution; reconcile this checklist after the update succeeds.
- Numbered math notes are review requests from the draft, not independently verified conclusions or approved implementation changes. Keep the original notes until their resolutions are reflected consistently in prose, equations, code, and results as applicable.
- Thesis-only and presentation-only TODOs are outside this paper checklist. Shared sections can affect those documents too.

## Outline and organization

- [ ] **Write the outline of the paper in the paper main page copy, using notes from my meeting with my instructor, and then migrate the paper there if needed.** Identify the intended copy and meeting notes when we work on this item; neither has been inferred here.
- [ ] Agree which formalism, methodology, results, and appendices belong in the final paper using that outline.

## Explicit TODOs and unfinished text

- [ ] Finalize the working title. Source: [main document](ddppaper2.tex), `TITLE WORK IN PROGRESS`.
- [ ] Write the abstract. Source: [main document](ddppaper2.tex), `TODO{Abstract}`.
- [ ] Write acknowledgments. Source: [main document](ddppaper2.tex), `TODO{acknowledge}`.
- [ ] Describe hyperparameter selection. Source: [model specs](../common/sections/methodology/model_specs.tex), explicit TODO.
- [ ] Choose the multidimensional figure combinations, including whether to retain detector effects/nuisance fitting throughout the 2D and higher-dimensional comparisons. Source: [multiple observables](../common/sections/results/multiple_dimensions.tex), explicit TODO.
- [ ] Write the conclusion, including the currently empty “Usability” and “Looking Forward” subsections. Source: [conclusion](../common/sections/conclusion.tex).
- [ ] Complete the unfinished “can be seen in Figure” sentence with the intended 2D significance figure reference. Source: [simulated data](../common/sections/results/simulated_data.tex).

## Joint review: notes 1, 2, 3, 18, and part of 7

- [x] Clarify in the formalism and Appendix A that equality concerns generated background means before Poisson sampling, detector effects, and injection; observed counts may differ.
- [x] Require the signal-only normalization constraint `integral f n_R = 0` in the main-text formalism, with an equation label.
- [x] Confirm the meaning of equal samples with Elad and inspect Poisson count generation in `data_tools/dataset_config.py`. This distinction was missing from the paper; it is now stated explicitly. Notes 3 and 18 remain applicable because observed counts can differ.
- [ ] Decide how the appendix should handle unequal-size samples: restrict the interpretation of the current pooled approximation or adopt and derive a consistent balanced-reference prescription. No likelihood or implementation change has been selected.
- [ ] Resolve note 2 separately: equality of sample sizes does not establish the stated detector-efficiency/asymmetry relation or supply missing intensity factors.
- [x] Correct note 7’s missing signal logarithm, retain the full parameter-dependent intensity integral, and correct the hypothesis reference. Trace the logarithms through Appendix A and correct its reference-only constant to a sum of logarithms. The final displayed loss and implemented logarithmic terms already have the correct form; no training-code change is needed for this correction.
- [ ] Review normalization of the combined signal/nuisance factors with note 5; the signal-only constraint is insufficient. Paper compilation succeeded (24 pages); the existing combined detector-equation reference remains undefined.

Working decision: treat these notes as one linked discussion, retain their source text for now, and record the clarified generation scope, signal-only constraint, and note 7 algebra correction as complete. No mathematical proposals in the notes are accepted solely by this status update.

## Mathematical review notes

Review these together before choosing changes. Source note numbers provide stable anchors even if equation and figure numbering changes.

- [ ] **Math note 1:** See the joint review above for partial progress and outstanding decisions. Review the balanced versus pooled reference definition and the normalization constraint on f. Source: [multiple_dimension_distributions](../common/sections/formalism/multiple_dimension_distributions.tex).
- [ ] **Math note 2:** See the joint review above for partial progress and outstanding decisions. Review the detector-asymmetry expression, efficiency normalization, event-count factors, and truth overlays. Source: [detector_affected_distributions](../common/sections/formalism/detector_affected_distributions.tex).
- [ ] **Math note 3:** See the joint review above for partial progress and outstanding decisions. Review null-model adequacy with unequal sample sizes; decide whether balanced weighting or a different likelihood is needed, and validate with asymmetric background-only toys. Source: [detector_affected_distributions](../common/sections/formalism/detector_affected_distributions.tex).
- [ ] **Math note 4:** State and validate control-region assumptions: signal contamination, SR/CR transfer, observable coverage, and sample independence; address omitted-variable effects. Source: [full_hypotheses](../common/sections/formalism/full_hypotheses.tex).
- [ ] **Math note 5:** Review closure of the combined signal/nuisance parametrization and its interpretation. Source: [full_hypotheses](../common/sections/formalism/full_hypotheses.tex).
- [ ] **Math note 6:** Evaluate the proposed conditional label-likelihood reformulation against the intended statistical model before choosing an approach. Source: [distribution_likelihood](../common/sections/formalism/distribution_likelihood.tex).
- [ ] **Math note 7:** The missing logarithm, parameter-dependent integral, hypothesis reference, and appendix cancellation explanation have been corrected. Final loss unchanged; combined-factor normalization remains open. Compilation succeeded with the existing malformed detector reference warning. See the joint review above. Source: [distribution_likelihood](../common/sections/formalism/distribution_likelihood.tex).
- [ ] **Math note 8:** Clarify conversion from the test statistic to significance for multiple parameters and an empirically calibrated null. Source: [distribution_likelihood](../common/sections/formalism/distribution_likelihood.tex).
- [ ] **Math note 9:** Check how the model guarantees a finite loss and document the actual output bounds. Source: [loss_mechanics](../common/sections/formalism/loss_mechanics.tex).
- [ ] **Math note 10:** Review applicability of Wilks/Wald and distinguish heuristic chi-square references from justified null distributions. Source: [hypothesis_space_bases](../common/sections/formalism/hypothesis_space_bases.tex).
- [ ] **Math note 11:** Review null calibration with active nuisance parameters; evaluate null-preserving resampling and validate calibration. Source: [hypothesis_space_bases](../common/sections/formalism/hypothesis_space_bases.tex).
- [ ] **Math note 12:** Review the injected-significance benchmark for a finite comparison sample and treatment of zero-background bins. Source: [significance_estimation](../common/sections/formalism/significance_estimation.tex).
- [ ] **Math note 13:** Specify empirical tail probabilities, finite-toy significance limits, median reporting, and quantile bands. Source: [significance_estimation](../common/sections/formalism/significance_estimation.tex).
- [ ] **Math note 14:** Reconcile the clamping description with the implemented bounded parametrization and boundary behavior. Source: [model_specs](../common/sections/methodology/model_specs.tex).
- [ ] **Math note 15:** Review the training algorithm: null/alternative initialization and best-fit retention, reference construction relative to injection, and identical non-convergence handling for null and signal toys. Source: [test_statistic_algorithm](../common/sections/methodology/test_statistic_algorithm.tex).
- [ ] **Math note 16:** Correct the interpretation of signal after shuffling and state when label exchangeability is valid. Source: [test_statistic_algorithm](../common/sections/methodology/test_statistic_algorithm.tex).
- [ ] **Math note 17:** Quantify agreement with the chi-square reference, reconcile degree counts in text and figures, and qualify convergence claims. Source: [reproduction](../common/sections/results/reproduction.tex).
- [ ] **Math note 18:** See the joint review above for partial progress and outstanding decisions. Reconcile the appendix integral approximation with the chosen reference weighting or likelihood. Source: [t_value_computation](../common/sections/appendices/t_value_computation.tex).
- [ ] **Math note 19:** Review the Fisher-information rank, null identifiability, and claimed asymptotic distribution; substantiate the accepted statements. Source: [borken_Wilks_Wald](../common/sections/appendices/borken_Wilks_Wald.tex).
- [ ] **Math note 20:** Reconcile the 2D chi-square degree count and document a fixed, comparable training budget. Source: [borken_Wilks_Wald](../common/sections/appendices/borken_Wilks_Wald.tex).

## Editorial and reproducibility follow-up

These items come from visible inconsistencies or sparse descriptions in the sources; their scope should follow the agreed outline.

- [ ] Remove overlapping introductory descriptions of collider searches and edit grammar and terminology. Source: [introduction](../common/sections/introduction.tex).
- [ ] Check the exponential background density’s sign, normalization, domain, and dimension-index range against the actual generator; document the signal parameters used in the figures. Source: [data and signals](../common/sections/results/datasets_and_signals.tex).
- [ ] Reconcile architecture, initialization, normalization, nuisance parametrization, optimizer, and training-budget descriptions with the configurations that produced the selected results. Sources: [model specs](../common/sections/methodology/model_specs.tex), [algorithm](../common/sections/methodology/test_statistic_algorithm.tex), [compute framework](../common/sections/methodology/compute_framework.tex).
- [ ] Specify CMS sample selection, SR/CR construction, observables, event counts/weights, and injected signals sufficiently to reproduce the comparisons. Source: [simulated data](../common/sections/results/simulated_data.tex).
- [ ] Reconcile figure captions and discussion with the selected plots, including 2D/4D projections, detector effects, nuisance fits, and significance calibration. Sources: shared results sections.
- [ ] Fix malformed references and punctuation, including the combined equation reference and unclosed parentheses in the detector-effects section. Source: [detector nuisance](../common/sections/results/detector_nuisance.tex).
- [ ] Verify all input files, image paths, labels, citations, and bibliography entries; compile the paper after the agreed edits. Check whether `nocite{*}` and the draft abbreviations/contents pages belong in the submission version. Source: [main document](ddppaper2.tex).
- [ ] Incorporate accepted math-note resolutions into the paper and remove the corresponding visible draft notes and TODO markers only after verification.

## PR description details to publish

No PR has been opened from this session: Git metadata writes are blocked. Include the following in the PR description when publishing becomes available.

Note 7: Correct the missing logarithm in the expanded signal log-likelihood, retain the full parameter-dependent Poisson intensity integral, and cite the nuisance-inclusive hypothesis. Trace the event-intensity product through Appendix A: signal logarithms survive in the alternative, whereas the shared reference-density sums of logarithms and fixed event-count terms cancel in the likelihood ratio. Correct the appendix’s reference-only constant accordingly. The final displayed loss and implemented logarithmic terms already have the corrected form, so this correction requires no training-code change. Paper compilation succeeded (24 pages), with the existing malformed combined detector-equation reference warning.

## Reference choice discussion (notes 3 and 18)

- [ ] Distinguish count prefactors from shape normalization: multiplying by the observed count alone does not normalize `(1 +/- theta) n_R`. Confirm whether separately normalized shapes were intended; these are not currently written in equation 7.
- [ ] Choose among a balanced reference with balanced integration weights, a pooled reference with count-dependent deformation factors and corresponding constraints, or separately normalized component shapes with a rederived likelihood. A conditional label likelihood is a further reformulation option (note 6).
- [ ] If restricting to equal realized observed counts instead, document the required sampling/equalization procedure and its statistical consequences; equal generation means alone are insufficient.

No reference or detector parametrization change has been selected in this discussion.
