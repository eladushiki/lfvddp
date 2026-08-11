# Codebase Map

Generated: 2026-08-11T08:47:30Z | Files: 277 | Described: 0/277
<!-- gsd:codebase-meta {"generatedAt":"2026-08-11T08:47:30Z","fingerprint":"a693e15d9d19ca1eff67328f52f331229040e6be","fileCount":277,"truncated":false} -->

### (root)/
- `.gitattributes`
- `.gitignore`
- `.gitmodules`
- `AGENTS.md`
- `lfvddp.def`
- `pytest.ini`
- `README.md`
- `requirements.txt`
- `setup.py`

### configs/
- `configs/__init__.py`
- `configs/x_validate.py`

### configs/cluster/
- `configs/cluster/basic_cluster_config.json`

### configs/dataset/
- `configs/dataset/basic_generated_dataset_config.json`
- `configs/dataset/basic_loaded_dataset_config.json`

### configs/detector/
- `configs/detector/basic_detector_config.json`

### configs/plot/
- `configs/plot/basic_plot_config.json`

### configs/train/
- `configs/train/basic_train_config.json`

### configs/user/
- `configs/user/basic_user_config.json`

### data_tools/
- `data_tools/__init__.py`
- `data_tools/CMS_open_data.py`
- `data_tools/data_generation.py`
- `data_tools/data_utils.py`
- `data_tools/dataset_config.py`
- `data_tools/profile_likelihood.py`

### data_tools/detector/
- `data_tools/detector/detector_config.py`
- `data_tools/detector/detector_effect.py`
- `data_tools/detector/error.py`

### data_tools/detector/efficiency/
- `data_tools/detector/efficiency/shapes.py`
- `data_tools/detector/efficiency/uncertainty.py`

### data_tools/event_generation/
- `data_tools/event_generation/__init__.py`
- `data_tools/event_generation/background.py`
- `data_tools/event_generation/distribution.py`
- `data_tools/event_generation/signal.py`
- `data_tools/event_generation/types.py`

### frame/
- `frame/__init__.py`
- `frame/aggregate.py`
- `frame/config_handle.py`
- `frame/file_structure.py`
- `frame/git_tools.py`
- `frame/module_retriever.py`
- `frame/submit.py`
- `frame/time_tools.py`

### frame/cluster/
- `frame/cluster/__init__.py`
- `frame/cluster/cluster_config.py`
- `frame/cluster/is_same_as_commit.py`
- `frame/cluster/walltime.py`

### frame/command_line/
- `frame/command_line/__init__.py`
- `frame/command_line/execution.py`
- `frame/command_line/handle_args.py`

### frame/context/
- `frame/context/__init__.py`
- `frame/context/execution_context.py`
- `frame/context/execution_products.py`
- `frame/context/run_descriptor.py`

### frame/file_system/
- `frame/file_system/__init__.py`
- `frame/file_system/image_storage.py`
- `frame/file_system/numpy_events.py`
- `frame/file_system/root_reader.py`
- `frame/file_system/textual_data.py`
- `frame/file_system/training_history.py`

### mattiasdata/
- `mattiasdata/dothestuff_divide.py`
- `mattiasdata/dothestuff.py`
- `mattiasdata/dothestuff.py~`
- `mattiasdata/fill_histograms_loop.py`
- `mattiasdata/fill_histograms.cpp`
- `mattiasdata/outputs`
- `mattiasdata/run_histograms.sh`
- `mattiasdata/utils.py`
- `mattiasdata/utils.py~`
- `mattiasdata/utils.pyc`

### mattiasdata/histograms/
- `mattiasdata/histograms/ee_vs_mm_Lep0Pt histograms.root`
- `mattiasdata/histograms/ee_vs_mm_Lep0Pt_histograms.root`
- `mattiasdata/histograms/ee_vs_mm_Lep1Pt histograms.root`
- `mattiasdata/histograms/ee_vs_mm_Lep1Pt_histograms.root`
- `mattiasdata/histograms/ee_vs_mm_MLL histograms.root`
- `mattiasdata/histograms/ee_vs_mm_MLL_histograms.root`
- `mattiasdata/histograms/em_vs_me_Lep0Pt histograms.root`
- `mattiasdata/histograms/em_vs_me_Lep0Pt_histograms.root`
- `mattiasdata/histograms/em_vs_me_Lep1Pt histograms.root`
- `mattiasdata/histograms/em_vs_me_Lep1Pt_histograms.root`
- `mattiasdata/histograms/em_vs_me_MLL histograms.root`
- `mattiasdata/histograms/em_vs_me_MLL_histograms.root`

### neural_networks/
- `neural_networks/__init__.py`
- `neural_networks/differentiating_model.py`
- `neural_networks/NPLM`
- `neural_networks/NPLM_adapters.py`
- `neural_networks/utils.py`

### neural_networks/weights/taylor_expansion_net/
- `neural_networks/weights/taylor_expansion_net/parameters.py`

### paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/__init__.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/README.md`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/usage.ipynb`

### paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/analyze/
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/analyze/extra_plots_yuval.ipynb`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/analyze/fix_plots_for_paper.ipynb`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/analyze/new_analysis_utils_copy.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/analyze/new_analysis_utils.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/analyze/new_plot_utils.py`

### paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/configs/
- *(75 files: 75 .json)*

### paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/config_table_1cp_run2.json`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/config_table_1cp.json`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/config_table_comparison.json`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/new_analysis_utils_table.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/new_make_jobs_tar+csv_table.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/new_setting_table.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/new_submit_yuval_NPLM_table.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/new_submit_yuval.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/table scripts/new_training_table.py`

### paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/copy_txt_to_csv.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_make_jobs_tar+csv.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_setting_copy.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_setting_exp.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_setting.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_submit_BR_Mcoll.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_submit_inbar_Mcoll.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_submit_inbar_NPLM.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_submit_inbar_resample.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_submit_inbar.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_training_copy.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_training_exp.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_training_resample.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/new_training.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/NNutils_symm.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/save_jobs_script.py`
- `paper_scripts/Learning New Physics from Data -- a Symmetrized Approach/train/save.py`

### paper_scripts/tbd/
- `paper_scripts/tbd/investigate_loss_landscape.ipynb`
- `paper_scripts/tbd/investigate_root_data.ipynb`
- `paper_scripts/tbd/train_and_plot.ipynb`

### paper_scripts/tbd/common/
- `paper_scripts/tbd/common/bibliography.tex`
- `paper_scripts/tbd/common/definitions.tex`
- `paper_scripts/tbd/common/detector_efficiency.tex`
- `paper_scripts/tbd/common/packages.tex`
- `paper_scripts/tbd/common/preamble.tex`

### paper_scripts/tbd/research_proposal/
- `paper_scripts/tbd/research_proposal/research_proposal.tex`

### paper_scripts/tbd/research_proposal/run_at_20250620_231054_of_single_train.py_on_commit_ba286_pid_2578962/
- `paper_scripts/tbd/research_proposal/run_at_20250620_231054_of_single_train.py_on_commit_ba286_pid_2578962/context.json`

### paper_scripts/tbd/thesis/
- `paper_scripts/tbd/thesis/thesis.tex`

### paper_scripts/tbd/thesis/sections/
- `paper_scripts/tbd/thesis/sections/adding_nuisances.tex`
- `paper_scripts/tbd/thesis/sections/error_propagation.tex`
- `paper_scripts/tbd/thesis/sections/event_counting.tex`
- `paper_scripts/tbd/thesis/sections/hypothesis_testing.tex`

### plot/
- `plot/__init__.py`
- `plot/carpenter.py`
- `plot/create_plots.py`
- `plot/plot_factory.py`
- `plot/plot_utils.py`
- `plot/plots.py`
- `plot/plotting_config.py`

### test/
- `test/__init__.py`
- `test/conftest.py`
- `test/environment.py`
- `test/test_aggregate.py`
- `test/test_basic.py`
- `test/test_datasets.py`
- `test/test_file_input.py`
- `test/test_profile_likelihood.py`
- `test/test_runtime_resources.py`
- `test/test_train.py`

### test/configs/cluster/
- `test/configs/cluster/resource_aware_cluster_config.json`

### test/configs/dataset/
- `test/configs/dataset/cms_open_dataset_json.json`
- `test/configs/dataset/cms_open_dataset_txt.json`
- `test/configs/dataset/disjoint_1D_generated_dataset_config.json`
- `test/configs/dataset/disjoint_2D_generated_dataset_config.json`

### test/configs/dataset/dataset_specs/
- `test/configs/dataset/dataset_specs/CMS_mc_RunIISummer20UL16NanoAODv9_VVTo2L2Nu_MLL-1toInf_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.json`
- `test/configs/dataset/dataset_specs/CMS_mc_RunIISummer20UL16NanoAODv9_VVTo2L2Nu_MLL-1toInf_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt`

### test/configs/detector/
- `test/configs/detector/basic_1D_detector_config.json`
- `test/configs/detector/basic_2D_detector_config.json`

### test/configs/train/
- `test/configs/train/long_1D_train_config_with_nuisance.json`
- `test/configs/train/long_1D_train_config_without_nuisance.json`
- `test/configs/train/profile_1D_train_config_with_nuisance.json`
- `test/configs/train/short_1D_train_config_with_nuisance.json`
- `test/configs/train/short_1D_train_config_without_nuisance_like_nplm.json`
- `test/configs/train/short_1D_train_config_without_nuisance.json`
- `test/configs/train/short_2D_train_config_with_nuisance.json`
- `test/configs/train/short_2D_train_config_without_nuisance.json`

### test/context/
- `test/context/test_execution_context.py`

### test/context/configs/
- `test/context/configs/mixed_generated_and_resampled_datasets.json`
- `test/context/configs/walltime_1_minute.json`
- `test/context/configs/walltime_3_minutes_1_minute_limit.json`
- `test/context/configs/walltime_73_hours.json`

### test/data_generation/
- `test/data_generation/test_data_generation.py`
- `test/data_generation/test_generator_modes.py`

### test/data_generation/configs/dataset/
- `test/data_generation/configs/dataset/generator_modes_dataset_config.json`
- `test/data_generation/configs/dataset/small_exact_sized_loaded_dataset_config.json`

### test/detector/
- `test/detector/test_detection.py`

### test/detector/configs/
- `test/detector/configs/detector_affected_basic_ds_2.json`
- `test/detector/configs/detector_affected_basic_ds.json`
- `test/detector/configs/detector_perfect_basic_ds.json`

### test/plot/
- `test/plot/test_create_plots.py`
- `test/plot/test_plot_factory.py`
- `test/plot/test_plot_utils.py`

### test/submission/
- `test/submission/submit_test_utils.py`

### test/submission/continuation/
- `test/submission/continuation/test_continuation.py`
- `test/submission/continuation/test_submit_continuation_server.py`

### test/submission/continuation/configs/
- `test/submission/continuation/configs/continuation_cluster_config.json`

### train/
- `train/__init__.py`
- `train/checkpoints.py`
- `train/cpu_runtime.py`
- `train/model_trainer.py`
- `train/runtime_resources.py`
- `train/single_train.py`
- `train/submit_train.py`
- `train/tensorboard_clutch.py`
- `train/train_config.py`
- `train/train_utils.py`
- `train/training_names.py`
- `train/training_profiler.py`
