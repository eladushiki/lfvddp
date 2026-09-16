# Issue 018 S01 Baseline Caller Inventory

## Scope and provenance

- Tested source revision: `c8cfbe4f6bc083185b68bddbeb3d361d4c5da530`.
- Branch: `codex/issue-018-configurable-signal-bases`; `HEAD...origin/main = 3 0` after guarded alignment.
- Environment: Python 3.11.16, PyTorch 2.11.0, NumPy 2.4.4, CPU, float64 model tensors, one Torch thread for numerical locks.
- Baseline fixtures: `baseline_1D_omitted_f_binned_nuisance.json`, `baseline_1D_omitted_f_disabled_nuisance.json`, and `baseline_1D_omitted_f_neural_nuisance.json`.
- Search was performed on the aligned source before function-space extraction. No secrets, event payloads, serialized checkpoints, or large raw search output are included here.

## Caller and entry-point matrix

| Concern | Current implementation and callers | Baseline coverage or disposition |
|---|---|---|
| File composition | `frame/file_system/textual_data.py`: recursive config discovery and ordered shallow merge; `frame/command_line/handle_args.py:create_config_from_paths`; `frame/context/execution_context.py`: context construction, serialization, reload, comparison, and continuation discovery | Existing config composition tests plus the three new indirect `function_execution_context` fixtures. New fixture files intentionally omit all future f-role fields. |
| Train schema and validation | `train/train_config.py:TrainConfig`, `train__nn_*`, `train__data_is_train_for_nuisances`, `train__nuisance_is_neural_network`, nuisance-bin fields and `_validate_nuisance_configuration`; `configs/x_validate.py` cross-config/backend validation | Existing `test_train.py` validation/config tests; new fixtures freeze legacy defaults and disabled/binned/neural nuisance resolution. New f schema/validator must remain orthogonal to these fields. |
| Model construction | `neural_networks/differentiating_model.py:DifferentiatingModel.__init__`, `_build_signal_hypothesis_estimator`, `_SignalRegionShiftEstimator`; `_build_nuisance_estimators` selects `BlankNuisanceEstimator`, `ScalarBinnedNuisanceEstimator`, or `NeuralPerEventNuisanceEstimator` | `test/test_signal_function_baseline.py` asserts ordered parameter names, shapes, state keys, and initial state digest for all three nuisance modes. Denominator construction asserts no signal shift network. |
| Numerator and denominator semantics | `DifferentiatingModel.forward`, `_signal_region_shift`, `_signal_shift_log_terms`, loss assembly, and static loss methods. Numerator owns signal-region `f`; denominator omits it rather than selecting a disabled f family. | New baseline exercises numerator loss and denominator primary/theta relationship. Existing static-loss, compact-loss, and no-nuisance tests remain regression coverage. |
| Nuisance preparation/evaluation | `neural_networks/nuisance_calculation.py:NuisanceCalculation` interface; `BlankNuisanceEstimator`, `ScalarBinnedNuisanceEstimator`, `NeuralPerEventNuisanceEstimator`; `data_tools/detector/detector_effect.py` supplies bin geometry | Existing compressed-binned and neural-preparation tests; new baseline covers parameter ownership, predictions, strict restore, and one-step behavior for binned, disabled, and neural modes. |
| Initialization and optimization | `DifferentiatingModel._initialize_parameters`, `configure_optimizers`, `_set_learning_rate_for_epoch`, training loop and `nuisance_calculation.clamp_parameters` | New deterministic one-step loss/state/prediction locks; existing learning/convergence/training-profile tests cover public training behavior. Initialization must preserve omitted adaptive f bitwise. |
| LFVDDP launcher and worker dispatch | `train/model_trainer.py:TrainLauncher`, training model construction and fit/checkpoint paths; `frame/submit.py`, `frame/aggregate.py`, `frame/command_line/execution.py` dispatch/aggregation paths | Existing `test/test_train.py`, submission/continuation tests, and cluster dry-run coverage. Function-space resolution must be serialized into worker-visible config, not inferred from labels. |
| NPLM backend | `neural_networks/NPLM`, `neural_networks/NPLM_adapters.py`, launcher/backend selection and `train__like_NPLM` compatibility paths | N/A for f/nuisance implementation: NPLM remains a separate backend and must not appear in the shared family registry. Existing NPLM-specific tests and backend validation are the required regression surface; unsupported combinations must raise clearly. |
| Checkpoint save | `train/checkpoints.py:save_training_checkpoint`, `checkpoint_filename`, `_checkpoint_dir`, continuation checkpoint path helpers | New baseline asserts exact top-level keys (`model_name`, `epoch`, `model_state_dict`, `optimizer_state_dict`, `training_history`, `array_index`, `run_hash`) and state-key order. Checkpoint bytes are deliberately not hashed. |
| Checkpoint restore and continuation | `train/checkpoints.py:_torch_load`, `find_latest_training_checkpoint`; `DifferentiatingModel` checkpoint loading/fit continuation and `frame/context/execution_context.py` continuation discovery | New baseline performs `strict=True` state restore and continuation smoke for all three modes, including continued state/prediction digests. Existing public continuation tests cover discovery/current format. Legacy key translation and metadata versioning are implementation gates for later slices. |
| Primary prediction | `DifferentiatingModel.predict`, `_predict_ndf` and prediction processing/aggregation callers | New baseline hashes `predict` output with explicit label and checks `(9, 1)` shape. Existing model/prediction tests cover finite values and public invocation. |
| Secondary prediction | `DifferentiatingModel.predict_secondary` and plot/prediction processing callers | New baseline hashes `predict_secondary` separately and checks `(9, 1)` shape. Existing prediction-process tests remain regression coverage. |
| Nuisance prediction | `DifferentiatingModel.predict_theta`; neural network, binned values, and blank-zero branches | New baseline hashes `predict_theta`, checks `(9, 1)`, and verifies denominator semantics. Existing neural and no-nuisance tests cover shapes/zero behavior. |
| Execution products and metadata | `frame/context/execution_products.py`, `frame/context/run_descriptor.py`, `frame/aggregate.py`, training history/checkpoint metadata | Existing training-profile, history, aggregation, and continuation tests. Future resolved role specifications and rank metadata must be persisted without changing legacy checkpoint payload interpretation. |
| 1D and nD plotting | `plot/plot_factory.py`, `plot/plots.py`, `plot/plotting_config.py`, `plot/prediction_process.py` and n-dimensional prediction paths | `test/plot/test_prediction_process.py` retains 1D neural and 2D binned cases and adds 1D disabled nuisance. Plot utilities are not unit-tested per project rule; generation smoke asserts populated axes and valid hypothesis lines. |
| Plot specs and configuration | `plot/plotting_config.py`, tracked `configs/basic-generated/plot_config.json`, `configs/basic-loaded/plot_config.json`, and paper plot configs | Existing plot generation/config tests. No new function-space logic belongs in plot utilities; plots consume resolved model outputs/metadata. |
| Tracked basic config packs | `configs/basic-generated/*.json` and `configs/basic-loaded/*.json`, including train/detector/cluster/user/plot/dataset packs | Compatibility fixtures and later config migration tests. These are tracked inputs, not runtime-mutated test configs. |
| Tracked paper config packs | `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/configs/*.json` and table-script configs under `table scripts/` | Focused config/lifecycle fixture coverage is appropriate; full Cartesian execution is not required. Config migration must preserve intended modes and NPLM backend separation. |
| Test config packs | `test/configs/dataset`, `test/configs/detector`, `test/configs/train`, `test/configs/plot`, and `test/configs/cluster` | New baseline train packs use normal file composition; existing file-based fixtures are retained. No runtime config mutation is permitted. |
| Submission dry-run paths | `frame/submit.py`, `frame/cluster/*`, command-line execution/argument handling, and paper `new_submit_*.py` scripts | Existing server-marked tests are skipped unless explicitly enabled. No numerical baseline is applicable to scheduler submission; validate generated command/config payloads and preserve role specs in later integration tests. |

## Frozen legacy contract

The omitted f configuration resolves to the current two-layer adaptive sigmoid estimator: hidden `(4, 1)`, hidden bias `(4,)`, output `(1, 4)`, output bias `(1,)`, with numerator-only ownership. Binned nuisance owns one `(10,)` parameter; disabled nuisance owns no parameters; neural nuisance owns hidden `(2, 1)`, bias `(2,)`, output `(1, 2)`, bias `(1,)`. All three baseline modes use deterministic CPU float64 locks for initial state, representative loss, one-step predictions, strict state restore, and three-epoch continuation. Prediction digests distinguish `predict`, `predict_secondary`, and `predict_theta`.

## Risks and coupling to remove in later slices

- `DifferentiatingModel` currently contains role selection and shared lifecycle orchestration; extracting a canonical registry/factory must not duplicate family evaluation or let role options alias.
- Legacy config names and omitted defaults are resolved through `TrainConfig`; changing ownership without explicit translation risks silently changing old configs.
- Checkpoint state keys and optimizer parameter ordering are observable compatibility surfaces. Metadata/version translation must precede module ownership changes.
- NPLM selection is orthogonal to both learned roles; do not encode NPLM as a function-space family or reinterpret unsupported combinations.
- Geometry and family selection must be independently resolved for f and nuisance, including same-family and different-family combinations.

## Inventory commands

```bash
rg -n 'signal_region_shift|nuisance_is_neural|data_is_train_for_nuisances|NPLM|predict_theta|predict_secondary|save_training_checkpoint|find_latest_training_checkpoint' configs data_tools frame neural_networks plot train submission test --glob '*.py'
rg -n 'train__nn_|train__nuisance_|train__like_NPLM' configs test 'paper_scripts' --glob '*.json' --glob '*.py'
find configs test/configs 'paper_scripts/Learning New Physics from Data -- a Symmetrized Approach' -type f \( -name '*.json' -o -name '*.py' \)
```

This inventory is the pre-refactor caller assignment for S01/T02. Any later rename, extraction, or deletion must update this matrix and add or identify a meaningful enabled-feature test for each affected public path.
