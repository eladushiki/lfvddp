from types import SimpleNamespace

import torch

from frame.cluster.cluster_config import ClusterConfig
from frame.cluster.walltime import split_walltime
from frame.context.execution_context import (
    ExecutionContext,
    create_config_from_paramters,
)
from frame.context.run_descriptor import build_run_descriptor
from frame.file_structure import (
    SINGLE_TRAIN_SCRIPT_NAME,
    TRAINING_OUTCOMES_DIR_NAME,
)
from frame.file_system.textual_data import load_config_file
from test.environment import DEFAULT_CONFIG_PATHS
from train.checkpoints import find_latest_training_checkpoint, save_training_checkpoint
from train.submit_train import _replace_or_append_continue_from


def _cluster_config(walltime: str, walltime_limit: str = "72:00:00") -> ClusterConfig:
    return ClusterConfig(
        cluster__repo_url="git@example.com:owner/repo.git",
        cluster__environment_activation_command="",
        cluster__singularity_executable="singularity",
        cluster__qsub_queue="N",
        cluster__qsub_n_jobs=1,
        cluster__qsub_walltime=walltime,
        cluster__qsub_io=1,
        cluster__qsub_mem=2,
        cluster__qsub_ngpus_for_train=0,
        cluster__qsub_walltime_limit=walltime_limit,
    )


def _quick_termination_config(parent_dir):
    config_params = {}
    for config_path in DEFAULT_CONFIG_PATHS.values():
        config_params.update(load_config_file(config_path))
    config_params["cluster__qsub_walltime"] = "0:01:00"
    return create_config_from_paramters(
        config_params,
        out_dir=str(parent_dir),
        plot_in_place=True,
    )


def _specific_train_context(
    parent_dir,
    run_descriptor: str,
    array_index: int,
) -> ExecutionContext:
    return ExecutionContext(
        commit_hash="abc",
        config=_quick_termination_config(parent_dir),
        config_paths=[],
        command_line_args=[],
        run_descriptor=run_descriptor,
        array_index=array_index,
        is_debug_mode=True,
        is_no_build=True,
        is_only_train=True,
    )


def test_split_walltime_caps_jobs_at_72_hours():
    chunks = split_walltime("150:30:00")

    assert chunks == [
        "72:00:00",
        "72:00:00",
        "6:30:00",
    ]


def test_cluster_config_serves_walltime_chunks():
    config = _cluster_config("73:00:00")

    assert config.cluster__qsub_walltime == "72:00:00"
    assert config.cluster__qsub_total_walltime == "73:00:00"
    assert config.cluster__qsub_needs_continuation

    first_chunk = config.next_walltime_chunk()
    second_chunk = config.next_walltime_chunk(1)

    assert first_chunk == "72:00:00"
    assert second_chunk == "1:00:00"


def test_cluster_config_uses_configured_walltime_limit():
    config = _cluster_config("0:03:00", walltime_limit="0:01:00")

    assert config.cluster__qsub_walltime == "0:01:00"
    assert config.cluster__qsub_total_walltime == "0:03:00"
    assert config.cluster__qsub_walltime_chunks == [
        "0:01:00",
        "0:01:00",
        "0:01:00",
    ]
    assert config.cluster__qsub_needs_continuation


def test_replace_or_append_continue_from_normalizes_argument():
    assert _replace_or_append_continue_from(
        ["--debug", "--continue-from", "/host/results"],
        "/app/results",
    ) == ["--debug", "--continue-from", "/app/results", "--continue"]

    assert _replace_or_append_continue_from(
        ["--debug", "--continue-from=/host/results"],
        "/app/results",
    ) == ["--debug", "--continue-from=/app/results", "--continue"]

    assert _replace_or_append_continue_from(
        ["--debug"],
        "/app/results",
        include_continue_flag=False,
    ) == ["--debug", "--continue-from", "/app/results"]


def test_checkpoint_discovery_uses_single_latest_path_per_array_index(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    run_root = tmp_path / "submit-run"
    run_root.mkdir(parents=True)
    config = _quick_termination_config(run_root)

    first_run_descriptor = build_run_descriptor(
        stamp="first",
        dirsafe_runtag=config.config__dirsafe_runtag,
        entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
        pid=1,
    )
    first_context = _specific_train_context(run_root, first_run_descriptor, array_index=7)
    first_context.save_self_to_out_file()
    save_context = SimpleNamespace(
        training_outcomes_dir=first_context.unique_out_dir / TRAINING_OUTCOMES_DIR_NAME,
        array_index=7,
        run_hash="abc",
    )
    first_path = save_training_checkpoint(
        context=save_context,
        model_name="model_A",
        model=model,
        optimizer=optimizer,
        epoch=4,
        training_history={"loss": [1.0]},
    )

    second_run_descriptor = build_run_descriptor(
        stamp="second",
        dirsafe_runtag=config.config__dirsafe_runtag,
        entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
        pid=2,
    )
    second_context = _specific_train_context(run_root, second_run_descriptor, array_index=7)
    second_context.save_self_to_out_file()
    save_context.training_outcomes_dir = second_context.unique_out_dir / TRAINING_OUTCOMES_DIR_NAME
    second_path = save_training_checkpoint(
        context=save_context,
        model_name="model_A",
        model=model,
        optimizer=optimizer,
        epoch=5,
        training_history={"loss": [1.0, 0.5]},
    )

    load_context = SimpleNamespace(
        is_continue=True,
        continue_from=run_root,
        training_outcomes_dir=tmp_path / "unused" / "training_outcomes",
        array_index=7,
        config=config,
    )
    found_path, checkpoint = find_latest_training_checkpoint(load_context, "model_A")

    assert first_path != second_path
    assert found_path == second_path
    assert checkpoint["epoch"] == 5

    wrong_array_context = SimpleNamespace(
        is_continue=True,
        continue_from=run_root,
        training_outcomes_dir=tmp_path / "unused" / "training_outcomes",
        array_index=8,
        config=config,
    )
    assert find_latest_training_checkpoint(wrong_array_context, "model_A") is None
