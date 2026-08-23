from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from data_tools.data_utils import DataSet
from frame.command_line.handle_args import (
    create_config_from_paths,
    parse_config_from_args,
)
from frame.context.execution_context import (
    ExecutionContext,
    version_controlled_execution_context,
)
from frame.context.run_descriptor import (
    build_run_descriptor,
    context_glob_for_run,
    parse_run_descriptor,
    run_descriptor_matches,
)
from frame.file_structure import (
    CONFIGS_DIR_NAME,
    CONTEXT_FILE_NAME,
    SINGLE_TRAIN_SCRIPT_NAME,
    SUBMIT_TRAIN_SCRIPT_NAME,
)
from frame.file_system.textual_data import (
    load_config_file,
    load_dict_from_json,
)
from test.environment import ConfigType


def _config_for_out_dir(function_execution_context, out_dir: Path):
    return create_config_from_paths(
        function_execution_context.config_paths,
        out_dir=str(out_dir),
    )


def _context_args(
    continue_from=None,
    extra_time=None,
    epochs_target=None,
) -> Namespace:
    if continue_from is not None:
        return Namespace(
            continue_from=continue_from,
            debug=False,
            extra_time=extra_time,
            epochs_target=epochs_target,
        )
    return Namespace(
        debug=True,
        build_container=False,
        only_train=False,
        continue_from=None,
        extra_time=extra_time,
        epochs_target=epochs_target,
    )


def test_epochs_target_preserves_continuation_argument():
    _, args = parse_config_from_args(
        ["--continue", "saved-run", "--epochs-target", "10"]
    )

    assert args.continue_from == Path("saved-run")
    assert args.epochs_target == 10


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/context/configs/walltime_1_minute.json"
            ),
        }
    ],
    indirect=True,
)
def test_run_descriptor_build_parse_and_match(function_execution_context):
    config = function_execution_context.config
    dirsafe_runtag = config.config__dirsafe_runtag
    descriptor = build_run_descriptor(
        stamp="run_20260706_124102_0.123456",
        dirsafe_runtag=dirsafe_runtag,
        entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
        pid=12345,
    )

    parts = parse_run_descriptor(descriptor, dirsafe_runtag=dirsafe_runtag)

    assert parts.stamp == "run_20260706_124102_0.123456"
    assert parts.dirsafe_runtag == dirsafe_runtag
    assert parts.entrypoint == SUBMIT_TRAIN_SCRIPT_NAME
    assert parts.pid == 12345
    assert run_descriptor_matches(
        descriptor,
        entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
        dirsafe_runtag=dirsafe_runtag,
    )
    assert not run_descriptor_matches(
        descriptor,
        entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
        dirsafe_runtag=dirsafe_runtag,
    )
    assert not run_descriptor_matches(f"manual_{SUBMIT_TRAIN_SCRIPT_NAME}_context")
    assert context_glob_for_run(dirsafe_runtag, SUBMIT_TRAIN_SCRIPT_NAME).endswith(
        CONTEXT_FILE_NAME
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/context/configs/walltime_73_hours.json"
            ),
        }
    ],
    indirect=True,
)
def test_execution_context_persists_qsub_submission_state(
    tmp_path,
    function_execution_context,
):
    context = ExecutionContext(
        commit_hash="abc",
        config=_config_for_out_dir(function_execution_context, tmp_path),
        config_paths=function_execution_context.config_paths,
        command_line_args=[],
        is_debug_mode=True,
        is_build_container=False,
    )

    first_walltime = context.next_qsub_walltime_chunk()
    context.record_qsub_submission(first_walltime, "12345", context.unique_out_dir)
    context.save_self_to_out_file()

    loaded_context = ExecutionContext.naive_load_from_file(
        context.unique_out_dir / CONTEXT_FILE_NAME
    )

    assert loaded_context.unique_out_dir == context.unique_out_dir
    saved_context = load_dict_from_json(context.unique_out_dir / CONTEXT_FILE_NAME)

    assert loaded_context.qsub_submissions[0]["job_id"] == "12345"
    assert "qsub_walltime_chunks" not in saved_context
    assert saved_context["config"]["cluster__qsub_walltime"] == "73:00:00"
    assert "cluster__qsub_walltime_chunks" not in saved_context["config"]
    assert "cluster__qsub_total_walltime" not in saved_context["config"]

    second_walltime = loaded_context.next_qsub_walltime_chunk()
    loaded_context.record_qsub_submission(
        second_walltime, "12346", loaded_context.unique_out_dir
    )
    loaded_context.save_self_to_out_file()

    reloaded_context = ExecutionContext.naive_load_from_file(
        context.unique_out_dir / CONTEXT_FILE_NAME
    )

    assert [
        submission["chunk_index"] for submission in reloaded_context.qsub_submissions
    ] == [1, 2]
    assert reloaded_context.next_qsub_walltime_chunk() is None


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/context/configs/walltime_3_minutes_1_minute_limit.json"
            ),
        }
    ],
    indirect=True,
)
def test_execution_context_persists_qsub_walltime_limit(
    tmp_path,
    function_execution_context,
):
    context = ExecutionContext(
        commit_hash="abc",
        config=_config_for_out_dir(function_execution_context, tmp_path),
        config_paths=function_execution_context.config_paths,
        command_line_args=[],
        is_debug_mode=True,
        is_build_container=False,
    )

    assert context.next_qsub_walltime_chunk() == "0:01:00"
    context.record_qsub_submission("0:01:00", "12345", context.unique_out_dir)
    context.save_self_to_out_file()

    loaded_context = ExecutionContext.naive_load_from_file(
        context.unique_out_dir / CONTEXT_FILE_NAME
    )

    assert loaded_context.config.cluster__qsub_walltime_limit == "0:01:00"
    assert loaded_context.config.cluster__qsub_walltime_chunks == [
        "0:01:00",
        "0:01:00",
        "0:01:00",
    ]
    assert loaded_context.next_qsub_walltime_chunk() == "0:01:00"


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/submission/continuation/configs/continuation_cluster_config.json"
            ),
        }
    ],
    indirect=True,
)
def test_continuation_prepares_next_chunk_before_yield(
    tmp_path,
    function_execution_context,
):
    config = _config_for_out_dir(function_execution_context, tmp_path)
    context = ExecutionContext(
        commit_hash="abc",
        config=config,
        config_paths=function_execution_context.config_paths,
        command_line_args=["submit_train.py", "--configs"],
        run_descriptor=build_run_descriptor(
            stamp="submit",
            dirsafe_runtag=config.config__dirsafe_runtag,
            entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
            pid=1,
        ),
        is_debug_mode=True,
        is_build_container=False,
    )
    context.record_qsub_submission(
        "0:01:00",
        "12345",
        context.unique_out_dir,
    )
    context.close()

    with version_controlled_execution_context(
        config=None,
        config_paths=None,
        command_line_args=[
            SUBMIT_TRAIN_SCRIPT_NAME,
            "--continue",
            str(context.unique_out_dir),
        ],
        args=_context_args(continue_from=context.unique_out_dir),
    ) as continued_context:
        saved_context = load_dict_from_json(
            continued_context.unique_out_dir / CONTEXT_FILE_NAME
        )

        assert continued_context.qsub_walltime_chunk == "0:01:00"
        assert not continued_context.run_successful
        assert saved_context["qsub_walltime_chunk"] == "0:01:00"
        assert not saved_context["run_successful"]


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/context/configs/walltime_1_minute.json"
            ),
        }
    ],
    indirect=True,
)
def test_extra_time_extends_a_recorded_single_chunk_submission(
    tmp_path,
    function_execution_context,
):
    config = _config_for_out_dir(function_execution_context, tmp_path)
    context = ExecutionContext(
        commit_hash="abc",
        config=config,
        config_paths=function_execution_context.config_paths,
        command_line_args=["submit_train.py", "--configs"],
        run_descriptor=build_run_descriptor(
            stamp="submit",
            dirsafe_runtag=config.config__dirsafe_runtag,
            entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
            pid=1,
        ),
        is_debug_mode=True,
        is_build_container=False,
    )
    context.record_qsub_submission(
        "0:01:00",
        "12345",
        context.unique_out_dir,
    )
    context.close()

    with version_controlled_execution_context(
        config=None,
        config_paths=None,
        command_line_args=[
            SUBMIT_TRAIN_SCRIPT_NAME,
            "--continue",
            str(context.unique_out_dir),
            "--extra-time",
            "0:00:30",
        ],
        args=_context_args(
            continue_from=context.unique_out_dir,
            extra_time="0:00:30",
        ),
    ) as continued_context:
        assert continued_context.config.cluster__qsub_total_walltime == "0:01:30"
        assert continued_context.qsub_submissions[0]["job_id"] == "12345"
        assert continued_context.qsub_walltime_chunk == "0:00:30"


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/context/configs/walltime_1_minute.json"
            ),
        }
    ],
    indirect=True,
)
def test_extra_time_is_available_to_non_submit_continuations(
    tmp_path,
    function_execution_context,
):
    config = _config_for_out_dir(function_execution_context, tmp_path)
    context = ExecutionContext(
        commit_hash="abc",
        config=config,
        config_paths=function_execution_context.config_paths,
        command_line_args=[SINGLE_TRAIN_SCRIPT_NAME, "--configs"],
        run_descriptor=build_run_descriptor(
            stamp="train",
            dirsafe_runtag=config.config__dirsafe_runtag,
            entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
            pid=1,
        ),
        is_debug_mode=True,
        is_build_container=False,
    )
    context.close()

    with version_controlled_execution_context(
        config=None,
        config_paths=None,
        command_line_args=[
            SINGLE_TRAIN_SCRIPT_NAME,
            "--continue",
            str(context.unique_out_dir),
            "--extra-time",
            "0:00:30",
            "--epochs-target",
            "750000",
        ],
        args=_context_args(
            continue_from=context.unique_out_dir,
            extra_time="0:00:30",
            epochs_target=750000,
        ),
    ) as continued_context:
        assert continued_context.config.cluster__qsub_total_walltime == "0:01:30"
        assert continued_context.config.train__epochs == 750000
        assert continued_context.qsub_submissions == []


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/submission/continuation/configs/continuation_cluster_config.json"
            ),
        }
    ],
    indirect=True,
)
def test_find_stamped_run_context_skips_parent_pytest_context(
    tmp_path,
    function_execution_context,
):
    parent_config = _config_for_out_dir(function_execution_context, tmp_path)
    dirsafe_runtag = parent_config.config__dirsafe_runtag
    parent_context = ExecutionContext(
        commit_hash="abc",
        config=parent_config,
        config_paths=function_execution_context.config_paths,
        command_line_args=[],
        run_descriptor=build_run_descriptor(
            stamp="parent",
            dirsafe_runtag=dirsafe_runtag,
            entrypoint="pytest",
            pid=1,
        ),
        is_debug_mode=True,
        is_build_container=False,
    )
    parent_context.save_self_to_out_file()

    submit_context = ExecutionContext(
        commit_hash="abc",
        config=_config_for_out_dir(
            function_execution_context,
            parent_context.unique_out_dir,
        ),
        config_paths=function_execution_context.config_paths,
        command_line_args=[],
        run_descriptor=build_run_descriptor(
            stamp="child",
            dirsafe_runtag=dirsafe_runtag,
            entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
            pid=2,
        ),
        is_debug_mode=True,
        is_build_container=False,
    )
    submit_context.record_qsub_submission(
        "0:01:00", "12345", submit_context.unique_out_dir
    )
    submit_context.save_self_to_out_file()

    found_context = ExecutionContext.find_stamped_run_context(
        parent_context.unique_out_dir,
        parent_context.config.config__dirsafe_runtag,
        entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
        require_continuation=True,
    )

    assert found_context is not None
    assert found_context.unique_out_dir == submit_context.unique_out_dir
    assert found_context.qsub_submissions[0]["job_id"] == "12345"
    assert (
        ExecutionContext.find_stamped_run_context(
            submit_context.unique_out_dir,
            parent_context.config.config__dirsafe_runtag,
            entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
            require_continuation=True,
        ).unique_out_dir
        == submit_context.unique_out_dir
    )


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/context/configs/walltime_1_minute.json"
            ),
        }
    ],
    indirect=True,
)
def test_discover_run_contexts_finds_single_train_context(
    tmp_path,
    function_execution_context,
):
    parent_config = _config_for_out_dir(function_execution_context, tmp_path)
    dirsafe_runtag = parent_config.config__dirsafe_runtag
    parent_context = ExecutionContext(
        commit_hash="abc",
        config=parent_config,
        config_paths=function_execution_context.config_paths,
        command_line_args=[],
        run_descriptor=build_run_descriptor(
            stamp="parent",
            dirsafe_runtag=dirsafe_runtag,
            entrypoint="pytest",
            pid=1,
        ),
        is_debug_mode=True,
        is_build_container=False,
    )

    single_train_context = ExecutionContext(
        commit_hash="abc",
        config=_config_for_out_dir(
            function_execution_context,
            parent_context.unique_out_dir,
        ),
        config_paths=function_execution_context.config_paths,
        command_line_args=[],
        run_descriptor=build_run_descriptor(
            stamp="child",
            dirsafe_runtag=dirsafe_runtag,
            entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
            pid=2,
        ),
        is_debug_mode=True,
        is_build_container=False,
    )
    single_train_context.save_self_to_out_file()

    discovered_contexts = ExecutionContext.discover_run_contexts(
        parent_context.unique_out_dir,
        entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
        dirsafe_runtag=dirsafe_runtag,
    )

    assert [
        (context.unique_out_dir, path.parent) for context, path in discovered_contexts
    ] == [(single_train_context.unique_out_dir, single_train_context.unique_out_dir)]


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.DATASET.value: Path(
                "test/context/configs/mixed_generated_and_resampled_datasets.json"
            ),
            ConfigType.DETECTOR.value: Path(
                "test/configs/detector/basic_1D_detector_config.json"
            ),
            ConfigType.TRAIN.value: Path(
                "test/configs/train/short_1D_train_config_without_nuisance.json"
            ),
        }
    ],
    indirect=True,
)
def test_continuation_recreates_generated_and_resampled_datasets(
    tmp_path,
    data_batch_events,
    monkeypatch,
    function_execution_context,
):
    monkeypatch.delenv("PBS_ARRAY_INDEX", raising=False)
    numpy_path = tmp_path / "events.npy"
    np.save(numpy_path, np.arange(40, dtype=float).reshape(-1, 1))

    fixture_descriptor = parse_run_descriptor(
        function_execution_context.run_descriptor,
        dirsafe_runtag=function_execution_context.config.config__dirsafe_runtag,
    )
    assert fixture_descriptor is not None

    initial_config_paths = function_execution_context.config_paths
    initial_command_line_args = [fixture_descriptor.entrypoint]
    with version_controlled_execution_context(
        config=_config_for_out_dir(function_execution_context, tmp_path),
        config_paths=initial_config_paths,
        command_line_args=initial_command_line_args,
        args=_context_args(),
    ) as initial_context:
        with monkeypatch.context() as temporary_working_directory:
            temporary_working_directory.chdir(tmp_path)
            initial_events = data_batch_events(initial_context)
        initial_run_dir = initial_context.unique_out_dir
        initial_descriptor = parse_run_descriptor(
            initial_context.run_descriptor,
            dirsafe_runtag=initial_context.config.config__dirsafe_runtag,
        )
        assert initial_descriptor is not None
        for category in (
            DataSet.DataSetCategory.A_CR,
            DataSet.DataSetCategory.B_CR,
        ):
            assert initial_events[category].shape == (8, 1)
            assert not np.array_equal(
                initial_events[category],
                np.arange(8, dtype=float).reshape(-1, 1),
            )

    with version_controlled_execution_context(
        config=None,
        config_paths=None,
        command_line_args=[
            initial_descriptor.entrypoint,
            "--continue",
            str(initial_run_dir),
        ],
        args=_context_args(continue_from=initial_run_dir),
    ) as continuation_context:
        with monkeypatch.context() as temporary_working_directory:
            temporary_working_directory.chdir(tmp_path)
            continuation_events = data_batch_events(continuation_context)

        assert continuation_context.unique_out_dir == initial_run_dir
        assert continuation_context.random_seed == 314159
        assert continuation_context.array_index is None
        assert (
            parse_run_descriptor(
                continuation_context.run_descriptor,
                dirsafe_runtag=continuation_context.config.config__dirsafe_runtag,
            ).pid
            == initial_descriptor.pid
        )
        assert continuation_context.config_paths == initial_config_paths
        assert continuation_context.command_line_args == initial_command_line_args
        assert initial_events.keys() == continuation_events.keys()
        for category in initial_events:
            np.testing.assert_array_equal(
                continuation_events[category],
                initial_events[category],
            )


@pytest.mark.parametrize(
    "function_execution_context",
    [
        {
            ConfigType.CLUSTER.value: Path(
                "test/context/configs/walltime_1_minute.json"
            ),
        }
    ],
    indirect=True,
)
def test_child_context_loading_matches_array_index(
    tmp_path,
    function_execution_context,
):
    config = _config_for_out_dir(function_execution_context, tmp_path)
    contexts = []
    for array_index, random_seed in ((1, 101), (2, 202)):
        context = ExecutionContext(
            commit_hash="abc",
            config=_config_for_out_dir(function_execution_context, tmp_path),
            config_paths=[],
            command_line_args=[],
            run_descriptor=build_run_descriptor(
                stamp=f"array_{array_index}",
                dirsafe_runtag=config.config__dirsafe_runtag,
                entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
                pid=array_index,
            ),
            array_index=array_index,
            random_seed=random_seed,
            is_debug_mode=True,
            is_build_container=False,
        )
        context.save_self_to_out_file()
        contexts.append(context)

    selected = ExecutionContext.load_child_run_context(
        parent_directory=tmp_path,
        entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
        array_index=2,
    )

    assert selected.random_seed == 202
    assert selected.unique_out_dir == contexts[1].unique_out_dir


def test_continuation_accepts_overrides_and_the_optional_debug_flag(capsys):
    config_paths, args = parse_config_from_args(["--continue", "results/run"])

    assert config_paths is None
    assert args.continue_from == Path("results/run")
    assert not args.debug

    config_paths, args = parse_config_from_args(
        ["--continue", "results/run", "--debug"]
    )

    assert config_paths is None
    assert args.continue_from == Path("results/run")
    assert args.debug

    config_paths, args = parse_config_from_args(
        [
            "--continue",
            "results/run",
            "--extra-time",
            "24:00:00",
            "--epochs-target",
            "750000",
            "--debug",
        ]
    )

    assert config_paths is None
    assert args.extra_time == "24:00:00"
    assert args.epochs_target == 750000
    assert args.debug

    _, args = parse_config_from_args(
        ["--continue", "results/run", "--epochs-target", "750000"]
    )
    assert args.epochs_target == 750000

    with pytest.raises(SystemExit):
        parse_config_from_args(["--continue", "results/run", "--build-container"])
    with pytest.raises(SystemExit):
        parse_config_from_args(["--continue", "results/run", "--extra-time", "24"])
    with pytest.raises(SystemExit):
        parse_config_from_args(["--continue", "results/run", "--epochs-target", "0"])
    with pytest.raises(SystemExit):
        parse_config_from_args([
            "--configs",
            "configs/basic-loaded/user_config.json",
            "--extra-time",
            "24:00:00",
        ])
    with pytest.raises(SystemExit):
        parse_config_from_args([
            "--configs",
            "configs/basic-loaded/user_config.json",
            "--epochs-target",
            "750000",
        ])
    capsys.readouterr()


@pytest.mark.parametrize(
    "config_pack", ("configs/basic-loaded", "configs/basic-generated")
)
def test_basic_config_packs_create_complete_configurations(config_pack):
    config_paths, _ = parse_config_from_args(["--configs", config_pack])
    config = create_config_from_paths(config_paths)

    assert config.train__epochs > 0
    assert config.detector__number_of_dimensions > 0


def test_config_directories_only_expand_supported_files(tmp_path):
    configs_directory = tmp_path / CONFIGS_DIR_NAME
    nested_directory = configs_directory / "nested"
    nested_directory.mkdir(parents=True)
    json_path = configs_directory / "first.json"
    yaml_path = nested_directory / "second.yaml"
    json_path.write_text("{}")
    yaml_path.write_text("{}")
    (configs_directory / ".hidden").write_text("not a config")
    (configs_directory / "notes.txt").write_text("not a config")
    child_run_directory = tmp_path / "run_0000"
    child_run_directory.mkdir()
    (child_run_directory / "context.json").write_text("{}")

    config_paths, _ = parse_config_from_args(["--configs", str(tmp_path)])

    assert config_paths == [json_path, yaml_path]


def test_missing_config_path_is_reported_before_format_detection(tmp_path):
    missing_path = tmp_path / "temporarily-unavailable-configs"

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        parse_config_from_args(["--configs", str(missing_path)])


def test_unsupported_config_error_identifies_extensionless_path(tmp_path):
    extensionless_path = tmp_path / ".config"
    extensionless_path.write_text("{}")

    with pytest.raises(ValueError, match=r"<none>.*\.config"):
        load_config_file(extensionless_path)
