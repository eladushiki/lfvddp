"""Composition checks for the background-only Issue 018 config packs."""

from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path

from frame.command_line.handle_args import create_config_from_paths
from frame.context.execution_context import ExecutionContext, create_config_from_paramters
from frame.file_system.textual_data import load_config_params_from_paths
from frame.cluster.cluster_config import ClusterConfig
from data_tools.dataset_config import DatasetConfig
from data_tools.detector.detector_config import DetectorConfig
from frame.config_handle import UserConfig
from plot.plotting_config import PlottingConfig
from train.function_space_config import FunctionSpaceFamily, RoleState, TrainingBackend
from train.train_config import TrainConfig


CONFIG_CLASSES = (
    ClusterConfig,
    DatasetConfig,
    DetectorConfig,
    PlottingConfig,
    TrainConfig,
    UserConfig,
)
KNOWN_CONFIG_KEYS = {
    field.name
    for config_class in CONFIG_CLASSES
    for field in fields(config_class)
}

CONFIG_ROOT = Path(__file__).parent / "configs" / "background-only"
COMMON_ROOT = CONFIG_ROOT / "common"
MODES = (
    "adaptive_neural",
    "bin_indicators",
    "cubic_bspline",
    "fixed_sigmoid",
    "orthogonal_polynomial",
)


def _paths_for_mode(mode: str) -> list[Path]:
    return sorted(COMMON_ROOT.glob("*.json")) + [
        CONFIG_ROOT / mode / "train_config.json"
    ]


def _assert_no_unknown_keys(paths: list[Path]) -> None:
    for path in paths:
        with path.open() as stream:
            values = json.load(stream)
        unknown = set(values) - KNOWN_CONFIG_KEYS
        assert not unknown, f"{path} has unknown config key(s): {sorted(unknown)}"


def test_background_only_packs_compose_without_mutation():
    common_paths = sorted(COMMON_ROOT.glob("*.json"))
    assert len(common_paths) == 5

    common_keys: set[str] = set()
    for path in common_paths:
        with path.open() as stream:
            values = json.load(stream)
        assert common_keys.isdisjoint(values), f"duplicate common key in {path}"
        common_keys.update(values)

    for mode in MODES:
        paths = _paths_for_mode(mode)
        _assert_no_unknown_keys(paths)
        params = load_config_params_from_paths(paths)
        before = deepcopy(params)

        # Exercise both the normal path loader and the constructor used by the
        # execution-context fixture.  Config validation must not mutate inputs.
        create_config_from_paramters(params)
        assert params == before
        config = create_config_from_paths(paths)
        context = ExecutionContext(
            commit_hash="config-test",
            config=config,
            config_paths=paths,
            command_line_args=["pytest"],
        )

        resolved = context.config.train__resolved_function_space_config
        assert resolved.backend is TrainingBackend.LFVDDP
        assert resolved.f.family is FunctionSpaceFamily(mode)
        assert resolved.f.state is RoleState.ENABLED
        assert resolved.nuisance.family is FunctionSpaceFamily.BIN_INDICATORS
        assert resolved.nuisance.state is RoleState.ENABLED

        definitions = context.config.dataset__definitions
        assert definitions
        assert all(definition["type"] == "generated" for definition in definitions)
        assert all(
            definition["dataset__mean_number_of_signal_events"] == 0
            for definition in definitions
        )
        assert all("dataset__signal_generator" not in definition for definition in definitions)


def test_basic_packs_explicitly_select_adaptive_f():
    for pack_name, dataset_name in (
        ("basic-generated", "generated_dataset_config.json"),
        ("basic-loaded", "loaded_dataset_config.json"),
    ):
        paths = sorted((Path(__file__).parents[1] / "configs" / pack_name).glob("*.json"))
        config = create_config_from_paths(paths)
        assert config.train__f["family"] == "adaptive_neural"
        assert config.train__f["options"] == {
            "input_dimension": 2,
            "hidden_layer_nodes": 4,
        }
        assert dataset_name in {path.name for path in paths}
