# Decisions Register

<!-- Append-only. Never edit or remove existing rows.
     To reverse a decision, add a new row that supersedes it.
     Read this file at the start of any planning or research phase. -->

| # | When | Scope | Decision | Choice | Rationale | Revisable? | Made By |
|---|------|-------|----------|--------|-----------|------------|---------|
| D001 | Retrospective backfill from implemented CLI and configuration pipeline | configuration architecture | Configuration composition | Accept untyped JSON or YAML configuration fragments, expand directories recursively, then shallow-merge them in caller order before cross-configuration and validation. | This makes a run configurable through composable domain-specific fragments while preserving one merged configuration as the source of truth. | Yes | agent |
| D002 | Retrospective backfill from implemented execution pipeline | reproducibility | Execution context ownership | Run entry points inside a version-controlled execution context that records run metadata and seeds the relevant random backends. | Scientific comparisons need outputs to retain the configuration, provenance, and seed needed to reproduce or continue a run. | Yes | agent |
| D003 | Retrospective backfill from prior living decision register | scientific use | Interpretation of high-significance Z | Use high Z to prioritize follow-up theory and experimental investigation, not to claim discovery. | The method provides model-agnostic evidence of a discrepancy and is intended to direct theory development. | Yes | human |
| D004 | Retrospective backfill from prior living decision register | theory implementation | Paper-equation equivalence | Keep code mathematically equivalent to the paper's equations; permit implementation choices not specified there to evolve. | The paper is the authoritative mathematical formulation, while implementation remains an engineering concern. | Yes | human |
| D005 | PR #111 neural nuisance refactor | architecture | Encapsulate nuisance parameter modes behind a dedicated nuisance-calculation class hierarchy. | Use a common nuisance calculation interface with scalar-bin and neural per-event implementations; have DifferentiatingModel delegate nuisance preparation, evaluation, and loss contribution to it. | User directed a readability and duplication refactor after divergent scalar and neural nuisance paths accumulated in DifferentiatingModel. Centralizing mode-specific binning and evaluation keeps one model loss assembly and a single source of truth for nuisance behavior. | Yes | human |
| D006 | Requested loss-clarity refactor | architecture | Training-loss ownership | Centralize the complete negative log-likelihood in DifferentiatingModel._assemble_loss; nuisance implementations return only nuisance values and control-region weights. | A single, documented loss implementation keeps the paper-to-code mapping inspectable and prevents scalar and neural nuisance modes from duplicating mathematical terms. | Yes | human |
| D007 | Issue 010 performance follow-up | training arithmetic | Category grouping and linear SR reduction | Preserve the prepared A-then-B event order as contiguous slices and reduce the reciprocal linear SR integral to scalar sums plus one dot product. Represent neural CR categories as unweighted contiguous values and scalar bins as multiplicity-weighted values behind the same loss assembly. | Profiling showed boolean indexing and materialized full-event intermediates dominating CPU time while the nuisance network itself was inexpensive. The compact representation restores the earlier arithmetic efficiency without changing the loss or its gradients. | Yes | human |
| D008 | Issue 016 convergence follow-up | neural likelihood parameterization | Smooth bounded neural outputs | Parameterize neural signal shifts and neural nuisance values as the shared log-safe bound multiplied by `tanh` of the raw network output; retain explicit parameter clamping only for scalar-binned nuisances. | A hard output clamp has exactly zero gradient beyond its boundary and trapped 11 observed trainings from their first checkpoint onward. A smooth parameterization preserves the mathematical bounds while retaining a recovery gradient. | Yes | human |
| D009 | PBS thread-warning investigation | runtime measurement | Isolate loaded CPU thread pools experimentally | Use an opt-in six-case array that changes OpenMP, OpenBLAS, Torch capacity, and process topology one variable at a time; measure the whole PBS session with a separate one-thread `/proc` sampler whose contribution is excluded from results. | The live processes load Torch, GNU OpenMP, OpenBLAS, and jemalloc, while ptrace restrictions prevent stack attribution. Controlled differences can identify fixed and scalable thread contributions without adding persistent threads to the training processes. | Yes; remove the probe after selecting the production cap | human |
| D010 | Temporary thread-pool probe setup | cluster working norm | Initialize every new cluster checkout with the repository's documented setup script and activate that checkout's own environment before running project commands. | Each checkout's `.venv` is tied to its project path and locked dependencies. Recording this in the cluster-access skill avoids rediscovering setup instructions and prevents ad-hoc cross-checkout environment binds. | Yes | human |
| D011 | PBS active-thread validation | cluster runtime policy | Keep native OpenMP/OpenBLAS pools at one thread and set the Python coordinator's Torch pool to one before spawning parallel branch workers. | The current 8-CPU baseline reached 14 runnable threads; the controlled one-thread native-pool case reduced runnable demand to 7 while retaining the intentional Torch worker split. | Yes | human |
| D014 | PR #145 hypothesis-space dof | statistical reporting | Derive the analytic chi-square degree count from the configured signal hypothesis space and its fixed constraints, never from a sampled batch. | Wilks degrees describe the dimension of the alternative model, so Monte Carlo occupancy and detector fluctuation cannot change them. Concrete fixed families own their parameter dependencies and observed-count constraint; aggregation only reloads the saved context configuration. | Yes | human |
| D015 | PR #145 function-space simplification | configuration and model architecture | Use `train__backend` with a required `f` specification and optional/null nuisance specification as the only canonical role format. | One immutable structural format removes competing backend and role state switches. Both enabled roles then share one catalog and per-event module contract; bin indicators are an ordinary Cartesian linear feature space. | Yes | human |
| D016 | 2026-10-07 diagnostic-reference regression | plotting | Restore unconditional positive-degree chi-square references in the t-distribution and percentile-progression plots, using one shared diagnostic-count helper. | Adaptive LFVDDP uses signal-network parameters minus the observed-count constraint, NPLM retains its raw parameter count, and fixed families retain their constrained dimension. The requested calibration change applies to multi-run significance plots; it must not suppress these training diagnostics. This supersedes D014 only for diagnostic plot references. | Yes | human |
| D017 | PR #188 review | degree-count ownership | Remove the plot-specific count helper and centralize diagnostic counts in the existing effective_test_statistic_degrees_of_freedom function. | Both diagnostic plots use ResultAggregator to derive the common count from saved run contexts. This preserves one implementation and detects incompatible submissions; multi-run significance remains empirically calibrated. Supersedes D016's helper ownership. | Yes | human |

| D018 | 2026-10-08 | Cluster coordination | Keep submission state only in the configured canonical cluster checkout; serialize read/action/save transactions with an atomic directory lock. | All agents use the committed lock helper and atomically publish YAML while holding the lock; abandoned locks require verified manual recovery. | Shared state avoids divergent local queues; locking the side effect as well as the write prevents duplicate submissions. Directory creation works across shared-filesystem clients without relying on local advisory-lock settings. | Yes | human + agent |

| D019 | 2026-10-08 | Generated significance | Apply nominal detector acceptance to both generated signal and background densities in the analytic significance benchmark. | Reuse the configured detector efficiency and generator PDFs; integrate the full accepted likelihood contribution so the accepted signal subtraction and event loss are included in every dimension. Share the efficiency adapter between plotting and saved-run aggregation, and include detector settings in aggregation cache keys. | The generated-level axis concealed detector acceptance losses. This benchmark assumes the nominal response is known; uncertainty, smearing, and profiling require a separate treatment. Loaded datasets retain their existing binned calculation. | Yes | human + agent |

| D020 | 2026-10-09 | Signal-yield calibration | Include the configured nominal detector efficiency when inverting detector-level injected significance. | Add a dataset/detector configuration mode to the calibration CLI and reuse the forward likelihood and efficiency adapter. Read only calibration inputs so older cluster run packs do not need a current training specification. Explicit PDF/generator APIs retain backward-compatible generated-level defaults and accept an efficiency callable. | Plot benchmarks and requested signal-yield targets must use identical detected densities. This extends D019 to inverse calibration without changing uncertainty or smearing semantics. | Yes | human + agent |
| D021 | 2026-10-07 configurable neural depth | neural architecture and diagnostic counts | LFVDDP accepts an integer or a list of hidden-layer widths, including an empty list, independently for signal and neural nuisance roles. Count all signal-network weights and biases for diagnostic chi-square references without subtracting an observed-count constraint. | One architecture definition owns validation and parameter counting at every depth. Preserve single-hidden-layer initialization and checkpoint keys, scalar smoothly bounded outputs, and sigmoid hidden activations. NPLM remains integer-only with its existing raw count. This supersedes D016's adaptive diagnostic-count convention; parameter counts do not establish Wilks validity or replace empirical significance calibration. | Yes | human |
| D022 | 2026-10-08 neural dimension inference | configuration ownership | Remove neural input/output-width configuration. Infer input width from detector observables for construction and diagnostic counting; use one shared scalar-output constant. | Only hidden widths belong to architecture configuration. Discard obsolete neural dimension hints during resolution so saved contexts remain readable. Keep derived input geometry in checkpoint compatibility metadata to retain single-layer checkpoint compatibility. NPLM retains integer-only hidden widths. | Yes | human |
| D023 | 2026-10-09 neural factory review | family interface ownership | Pass observable names and counts through common family interfaces; the adaptive class owns required-observable validation and architecture counting. | Factories dispatch uniformly without neural-family branches. Deterministic families retain geometry-based construction and degree counts. | Yes | human |
| D024 | 2026-10-16 array archival acceptance | Cluster routine | Define array acceptance once in the submission-state schema and reuse it for completion, plotting, and archival. | Accepted partial arrays can be archived after all plot dependencies complete; preserve failure/missing-output evidence and remaining artifacts instead of imposing a stricter archival threshold. Keep partial-array intermediate pruning disabled. | Array acceptance must not diverge between workflow stages when its threshold changes. | Yes | human |
| D025 | 2026-10-18 recovered-array cleanup | Cluster routine | Recognize complete walltime recovery from ordered saved array submissions while preserving historical failure logs. | Pruning requires every worker context and every latest-array element to succeed; only saved earlier walltime kills may be recovered. Remove only exit-zero PBS logs, retain historical kills unchanged, and record their paths for eventual archival. Retry deferred cleanup without regenerating analyzed plots. | A historical walltime kill does not invalidate a successful continuation, and its audit evidence must survive cleanup. This does not relax partial-array or active-job safety rules. | Yes | human + agent |

| D026 | 2026-10-18 multidimensional prediction visibility | plotting | Display multidimensional prediction results as separate aligned heatmaps with shared scales and external labels. | Preserve all observed/fitted density maps and all four model factors per region; add an unsmoothed A/B difference in Poisson counting-error units. Display model factors averaged over hidden grid coordinates so the neutral factor stays at unity. | Perspective and overlapping surfaces concealed localized bumps; independent maps remove occlusion, while the counting-error diagnostic separates coherent excesses from background count fluctuations. It is a visual diagnostic, not calibrated significance. The 1D presentation remains unchanged. | Yes | human + agent |

| D027 | 2026-10-18 prediction-layout follow-up | plotting | Minimize multidimensional figure whitespace with constrained layout and a font-sized run-stamp footer. | Replace fixed top/bottom margins and large nested-grid gaps with small physical padding; keep the 1D rendering and layout unchanged. | The user requested smaller top, bottom, and inter-panel gaps after reviewing the heatmap preview. The compact heatmap footer supersedes the fixed crop-row convention for this view only. | Yes | human |

### 2026-10-18: budgeted Adam/L-BFGS training

- Keep Adam as the compatibility default; opt into `adam_lbfgs` per run.
- Count full-data loss/gradient evaluations rather than L-BFGS outer iterations.
  Warmup uses floor(5% of the initial budget), capped at 2,000 updates.
- Treat each L-BFGS step as a checkpoint transaction. At budget exhaustion
  inside a line search, restore parameters and optimizer curvature to step
  entry while retaining every evaluated best candidate separately.
- Resume preserves the original warmup boundary and exact saved optimizer
  phase/state. Budget extension does not restart Adam. A completed truncated
  line search may be retried on extension, so extension need not reproduce
  a run originally configured with the larger budget.
- Forty thousand evaluations per hypothesis preserves the tested 2,000-update
  warmup. Neither this total budget nor the 5% split is an established optimum.
- Caller inventory: `DifferentiatingModel.fit`, `calc_min_LFVNN`, checkpoint
  load/save, `ModelTrainer` checkpoint-completion shortcut, sequential/parallel
  training launchers, CLI epoch-target continuation, static denominator, and
  primary/secondary/nuisance prediction. Neural modules and plot geometry are
  unchanged, so no corresponding 1D/ND plot-spec change is required.

### 2026-10-19: share the optimizer training loop

- Supersedes the separate hybrid-fit implementation: Adam and Adam/L-BFGS now
  use the same `fit` loop and the same closure for forward loss, gradients,
  best-model tracking, history, progress, and profiling.
- `HybridOptimizer` owns the warmup transition, evaluation counter, epoch
  iterator, underlying optimizers, and interrupted-line-search rollback.
  Checkpoint hooks restore its phase before loading optimizer tensors.
- Checkpoint writes stay outside the closure so that history boundaries crossed
  inside a line search never persist a partial optimizer transaction.
- Updated caller inventory: `_train_step` now receives the current epoch and
  evaluation callback; its three existing test replacements use that same hook.
  Real CPU worker integration covers hybrid training with learned nuisance.

### 2026-10-19: inherit the PyTorch optimizer protocol

- `HybridOptimizer` subclasses and initializes `torch.optim.Optimizer`. It
  inherits zeroing and state serialization/loading; registered native hooks
  add schedule metadata and rebind the active Adam/L-BFGS delegate after load.
- The facade and delegate share state and parameter-group objects. Switching
  algorithms or rolling back a line search updates those objects in place.
  Loading follows PyTorch's native replacement semantics, then rebinds the
  delegate before user post-load hooks run.
- Closures follow the standard zero-argument protocol. The common training
  closure reads the hybrid evaluation counter to label each loss evaluation.
- A single group is supported, matching L-BFGS. Additional groups are rejected
  at construction and via `add_param_group`. The transition resets learning
  rate to 1 and preserves `initial_lr` and parameter names. StepLR attachment
  and post-load operation are tested; arbitrary scheduler policies are not
  implied, and scheduler state remains separately owned by the caller.
- Caller inventory remains the shared fit/closure, checkpoint hooks, training
  launchers (including real CPU workers), predictions, plotting integration,
  and profiler. This changes optimizer integration rather than likelihood or
  model geometry, so 1D/ND plot specifications need no update.


### 2026-10-19: count final array elements when pruning relaunched workers

- Scheduler relaunches can leave `run_successful: false` contexts before work
  starts. Directory count alone cannot classify final array elements as failed.
- Permit a superseded launch only when it contains just `context.json`, records
  no products, and a later successful context covers the same valid array index.
  Require exactly one successful context for every expected element; preserve
  the superseded contexts. Require complete exit-zero latest-array PBS logs for
  initial arrays as well as continuations. Unknown or ambiguous evidence blocks
  deletion before any artifacts change.
- Caller inventory: `successful_worker_directories` and `verified_pbs_logs` feed
  only `prune_finished_intermediates`; public CLI selects explicit targets or
  `--all-under-root`, with dry-run, actual cleanup, and repeat behavior. Regression
  coverage exercises both selectors, ordinary and continued arrays, preservation
  of contexts/final statistics/plots/historical failures, and rejection cases.
  Archive/restore behavior stays covered by the existing helper suite. This
  changes artifact validation; training, prediction, model restore, and 1D/ND
  plotting semantics are unaffected.

| D028 | Plot09 resampled reference | significance plotting | Honor an explicitly supplied reference directory even when its saved dataset contains signal; warn with its signal configuration. Keep automatic discovery background-only. | The null for a resampling comparison is the user-selected empirical reference distribution. Signal content does not invalidate that explicitly chosen null. The shared performance plot handles every dataset dimension. | Yes | human |


| D029 | 2026-10-20 prediction bulk display | plotting | Offer explicit percentile display limits in both 1D and multidimensional prediction plots, keeping observable ranges independent. | Pool A/B and SR/CR before selecting each observable's percentiles. Keep full-data defaults, full sample normalization, and saved statistical histogram edges unchanged. | Extreme momentum tails concealed the populated region. Cropping is a display decision rather than an analysis selection. PR #207 review explicitly rejected forcing different observables onto a common numerical range. | Yes | human + agent |

### 2026-10-20: prediction display range caller inventory (D029)

- `data_tools.histogram_binning.display_edges_by_observable` is called by both
  prediction entry points, `train.single_train.save_training_data_samples`, and
  plot utility tests. Only plotting passes the new optional display settings;
  saved sample edges retain the full observed range for statistical use.
- Factory dispatch selects 1D for one observable and the multidimensional view
  otherwise. Both paths share binning and prediction-grid bounds. Existing
  prediction tests cover neural/disabled nuisance in 1D, binned nuisance in 2D,
  and projected grid evaluation. Model training/restore logic is unchanged.
- PR #207 review removed the cross-observable range option from configuration,
  binning, and both plotting callers. Preserve percentile limits independently
  for each observable; sharing limits between different physical quantities
  requires an explicit request. Saved-sample callers still use full-range defaults.
- Review validation: 42 prediction/factory/performance checks passed, including
  enabled nuisance and saved-sample binning callers. The 2D plot generation check
  renders successfully but fails its stale four-axes assertion (30 axes); this
  also reproduces with the original PR code. Manual checks cover full/cropped
  independent ranges, 1D/ND parity, finite/constant data, and invalid percentiles.
  Training, checkpoint restore, and model prediction implementations are unchanged.

### 2026-10-21: component percentile progression

- Numerator and denominator progression figures use separate explicit plot entries,
  enabled in both standard configs. Preserve the factory's one-entry/one-figure
  contract and allow custom configs to select outputs independently.
- All three quantities share the final-t quality selection and renderer. Component
  losses can be negative, so fit signed final-half limits with range padding;
  chi-square references and calibration lookup apply only to the test statistic.
- Caller inventory: the existing t entry point is selected through plot configs
  and `PlotFactory`; `create_configured_plots` persists each resulting figure.
  The two internal helpers also have direct utility-test callers, retained for
  compatibility. One dimension-independent renderer covers 1D and ND datasets.
  Training, checkpoint restore, and prediction entry points are unaffected.
- Validation: 42 existing utility, factory, and aggregation checks passed. Manual
  smoke checks rendered all three entry points in 1D/2D with an enabled nuisance
  config, paired final-t rejection, negative losses, constant denominators, and
  early transients. Component calibration lookup was forced to fail if accessed;
  neither component accessed it. Existing t curves, colors, and limits match
  the previous implementation. Signed-limit empty/constant cases also passed.


### 2026-10-21: common observable space before regional resampling

- Materialize components and advance loaded source pools as before, then
  project every completed category onto the ordered detector observable
  list before regional A/B finalization. Use the detector configuration as the
  single source of truth. Retain extra ROOT branches through loading/cuts;
  never invent values for cut-only columns in generated signal events.
- This fixes Plot09 when signal-region resampling is enabled: its signal-bearing
  A dataset was already projected while the background-only B dataset retained
  bookkeeping columns. Apply the same rule to control regions, zero-signal
  realizations, identity splitting, and resampling with or without replacement.
- Initialize sampled-component bookkeeping when constructing DataGeneration so
  direct category access works before get_batch. Preserve get_batch's per-batch
  reset, source depletion, sampled component contents, and final event counts.
- Caller inventory: DataGeneration.get_batch feeds train.single_train and shared
  training fixtures; __getitem__ feeds direct category callers; sampled_components
  feeds saved loaded-data significance samples. Regression coverage exercises both
  batch and category entry points, dimensions 1/2, reordered columns, all splitting
  modes, two successive source draws, exhaustion, detector integration,
  normalization, and training with enabled neural nuisance and disabled nuisance.
  Direct DatasetParameters.dataset__data and ROOT cut/loading behavior are unchanged
  and checked by their existing suites. Checkpoint/primary/secondary/nuisance
  prediction implementations and plot rendering are unchanged; existing training
  checks exercise model predictions/restore. No plotting parity change is needed.
- Existing sampling tests now explicitly select the committed 1D detector fixture
  matching their scalar NumPy input rather than inheriting unrelated ROOT columns.
- Validation: 100 data-generation, dataset, ROOT-input, and training checks passed,
  including the 18 new projection/category/training cases. The TensorFlow-only
  NPLM learning case was deselected because this macOS environment has no
  TensorFlow; the changed loaded-data path uses LFVNN/PyTorch. Ruff and whitespace
  checks passed. The original implementation failed the new regression before
  projection was added.

- Review clarified that observable compatibility is source-independent. Apply the
  projection unconditionally after component combination, without a loaded-data
  type check. Extra-observable generated fixtures cover batch/category access in
  1D/2D and reordered columns, preserving selected values and event counts. Source
  acquisition/sampling remains separate from the shared materialization invariant.
- Updated validation: 106 checks passed after removing the source-type condition;
  all six new generated-source cases failed on the previous loaded-only version.
  The same TensorFlow-only case remains excluded because its dependency is absent.

### 2026-10-22: matching signal zooms in two-dimensional prediction maps

- All 24 maps of an exactly 2D resonant-signal prediction figure receive the
  same inset window: the configured injected mean plus/minus three marginal
  standard deviations on each axis. Covariance entries are variances.
- Use configured signal geometry, with observable-name ordering, rather than
  estimating a center or width from noisy data or fitted model predictions.
  Multiple Gaussian injections use their enclosing window.
- Magnify the existing meshes and retain their color scales and binning; the
  inset changes presentation only. Background-only, nonresonant, 1D, and
  higher-dimensional projections do not receive insets. Their corresponding
  plot entry points require no change.
- Caller inventory: `plot_prediction_process_2d` calls
  `render_prediction_heatmaps`; the renderer also has direct plotting smoke
  callers. Its new optional bounds argument preserves those callers. No model,
  training, checkpoint, prediction-function, or statistical behavior changes.
