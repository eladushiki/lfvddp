# A/B dataset configuration caller inventory

- `DatasetConfig.load_dataset_parameters`, `dataset_parameters`, `get_parameters`,
  and `dataset__has_signal`: schema validation and deep-copy tests cover generated
  and loaded sources, fixed/Poisson counts, positive/unrestricted signals, and
  identity/resampling/replacement policies. Invalid fixtures cover missing,
  duplicate, explicit CR, legacy four-entry, missing-key, and unknown-type inputs.
- `configs.x_validate.cross_configure`: supplies dimensions and loaded signal
  observable names to both user definitions; derived CR inherits them. Loaded
  projection tests cover enabled signal overlays and resampling.
- `DataGeneration.__getitem__` and `get_batch`: existing generated/loaded tests
  exercise all four runtime categories. CR generation and ordered loaded sampling
  now assert zero signals and the copied SR background source/count.
- `SignalCalibrationConfig.signal_dataset_parameters`, calibration API and CLI:
  existing A/B calibration tests use two definitions. Derived CR entries do not
  become additional signal datasets.
- `ExecutionContext` creation, serialization, and continuation: replay tests use
  generated fixture pairs and loaded/resampled pairs with derived CR entries.
- `train.comparison_datasets.dataset_comparison_values` and `signal_curve_key`:
  share the schema validator without sampling; invalid fixtures test both runtime
  and comparison entry points.
- Training and downstream model integration: existing learning, function-space
  integration/checkpoint, and prediction-process tests cover 1D/2D, enabled nuisance
  models, primary/secondary/nuisance prediction, restore, and plot integration.
  Model implementation is unchanged.
- Plot consumers (`plots.py`, `plot_utils.py`) still receive four runtime parameter
  objects. No graph or plot specification changes are needed.
- Basic packs and the thread-pool probe contain only SR definitions. Historical
  paper scripts and CMS file-index fixtures without categories do not use this
  DatasetConfig schema and are unchanged.

## Validation

The selected data, calibration, training, function-space integration/checkpoint,
plot integration, context, and detector suites yielded 258 passing cases. The
final copy/schema test file separately passed all 24 cases after strengthening
nested-copy and zero-generated-signal assertions. Ruff formatting, Ruff F checks,
and `git diff --check` passed.

Five failures reproduce on a clean `origin/main` checkout: context tests
`test_execution_context_persists_qsub_submission_state`,
`test_execution_context_persists_qsub_walltime_limit`,
`test_continuation_prepares_next_chunk_before_yield`, and
`test_find_stamped_run_context_skips_parent_pytest_context`; and prediction-process
`test_prediction_process_plot_generation[2d-binned-nuisance]`, whose figure has
36 axes while the assertion expects four. That plot is generated before the
assertion fails. One NPLM learning case (`function_execution_context3`) cannot run
because TensorFlow is absent in the local macOS runtime; the same setup error
reproduces on main. These paths are explicit validation limitations, not changes
included in this feature.

Public dataset categories are `a` and `b`; the loader maps them to internal
A_SR/B_SR categories before constructing SR parameters and deriving CR copies.

After shortening the public categories to `a` and `b`, 180 targeted cases passed:
93 data/schema/calibration/detector cases and 87 training, model integration,
checkpoint, and context cases. Four known context failures were deselected in
that follow-up run. Internal category mapping and signal-free CR derivation
remain covered by the existing copy/materialization tests.
