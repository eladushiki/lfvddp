from data_tools.data_utils import DataSet
from data_tools.detector.detector_config import DetectorConfig
from frame.cluster.cluster_config import ClusterConfig
from plot.plotting_config import PlottingConfig
from train.train_config import NuisanceMode, TrainConfig


def test_requested_cluster_defaults():
    config = ClusterConfig("repo", "activate", "singularity", "N", 1, "01:00:00")
    assert config.cluster__qsub_io == 0.1
    assert config.cluster__qsub_mem == 2
    assert config.cluster__qsub_ncpus == 8
    assert config.cluster__qsub_ngpus_for_train == 0


def test_nuisance_mode_is_single_canonical_selector():
    config = TrainConfig(
        train__epochs=1,
        train__number_of_epochs_for_checkpoint=1,
        train__nn_inner_layer_nodes=2,
        train__nuisance_mode="neural",
        train__nuisance_nn_inner_layer_nodes=3,
    )
    assert config.nuisance_mode is NuisanceMode.NEURAL
    assert config.nuisances_enabled
    assert config.nuisance_uses_neural_network


def test_detector_effects_are_selected_by_dataset_family():
    config = DetectorConfig(
        detector__detect_observable_names=["x"],
        detector__binning_maxima=[1],
        detector__effects={"A": {"efficiency": "eff_a"}, "B": {"efficiency": "eff_b"}},
    )
    assert config.effects_for_category(DataSet.DataSetCategory.A_SR)["efficiency"] == "eff_a"
    assert config.effects_for_category(DataSet.DataSetCategory.B_CR)["efficiency"] == "eff_b"


def test_plotting_defaults_leave_plot_specifications_explicit():
    config = PlottingConfig("runs", [])
    assert config.plot__pyplot_styling["style.use"] == "classic"
    assert config.plot__figure_styling["plot"]["linewidth"] == 5
    assert config.plot__figure_size == (10, 9)
