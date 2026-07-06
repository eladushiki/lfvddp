from frame.context.execution_context import (
    ExecutionContext,
    create_config_from_paramters,
)
from frame.context.run_descriptor import (
    build_run_descriptor,
    context_glob_for_run,
    parse_run_descriptor,
    run_descriptor_matches,
)
from frame.file_structure import (
    CONTEXT_FILE_NAME,
    SINGLE_TRAIN_SCRIPT_NAME,
    SUBMIT_TRAIN_SCRIPT_NAME,
)
from frame.file_system.textual_data import load_config_file, load_dict_from_json
from test.environment import DEFAULT_CONFIG_PATHS


def _walltime_config(tmp_path, walltime: str, walltime_limit: str = "72:00:00"):
    config_params = {}
    for config_path in DEFAULT_CONFIG_PATHS.values():
        config_params.update(load_config_file(config_path))
    config_params["cluster__qsub_walltime"] = walltime
    config_params["cluster__qsub_walltime_limit"] = walltime_limit

    return create_config_from_paramters(
        config_params,
        out_dir=str(tmp_path),
        plot_in_place=True,
    )


def test_run_descriptor_build_parse_and_match(tmp_path):
    config = _walltime_config(tmp_path, "0:01:00")
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
    assert context_glob_for_run(dirsafe_runtag, SUBMIT_TRAIN_SCRIPT_NAME).endswith(CONTEXT_FILE_NAME)


def test_execution_context_persists_qsub_submission_state(tmp_path):
    context = ExecutionContext(
        commit_hash="abc",
        config=_walltime_config(tmp_path, "73:00:00"),
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        command_line_args=[],
        is_debug_mode=True,
        is_no_build=True,
    )

    first_walltime = context.next_qsub_walltime_chunk()
    context.record_qsub_submission(first_walltime, "12345", context.unique_out_dir)
    context.save_self_to_out_file()

    loaded_context = ExecutionContext.naive_load_from_file(context.unique_out_dir / CONTEXT_FILE_NAME)

    assert loaded_context.unique_out_dir == context.unique_out_dir
    saved_context = load_dict_from_json(context.unique_out_dir / CONTEXT_FILE_NAME)

    assert loaded_context.qsub_submissions[0]["job_id"] == "12345"
    assert "qsub_walltime_chunks" not in saved_context
    assert saved_context["config"]["cluster__qsub_walltime"] == "73:00:00"
    assert "cluster__qsub_walltime_chunks" not in saved_context["config"]
    assert "cluster__qsub_total_walltime" not in saved_context["config"]

    second_walltime = loaded_context.next_qsub_walltime_chunk()
    loaded_context.record_qsub_submission(second_walltime, "12346", loaded_context.unique_out_dir)
    loaded_context.save_self_to_out_file()

    reloaded_context = ExecutionContext.naive_load_from_file(context.unique_out_dir / CONTEXT_FILE_NAME)

    assert [submission["chunk_index"] for submission in reloaded_context.qsub_submissions] == [1, 2]
    assert reloaded_context.next_qsub_walltime_chunk() is None


def test_execution_context_persists_qsub_walltime_limit(tmp_path):
    context = ExecutionContext(
        commit_hash="abc",
        config=_walltime_config(tmp_path, "0:03:00", walltime_limit="0:01:00"),
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        command_line_args=[],
        is_debug_mode=True,
        is_no_build=True,
    )

    assert context.next_qsub_walltime_chunk() == "0:01:00"
    context.record_qsub_submission("0:01:00", "12345", context.unique_out_dir)
    context.save_self_to_out_file()

    loaded_context = ExecutionContext.naive_load_from_file(context.unique_out_dir / CONTEXT_FILE_NAME)

    assert loaded_context.config.cluster__qsub_walltime_limit == "0:01:00"
    assert loaded_context.config.cluster__qsub_walltime_chunks == [
        "0:01:00",
        "0:01:00",
        "0:01:00",
    ]
    assert loaded_context.next_qsub_walltime_chunk() == "0:01:00"


def test_find_stamped_run_context_skips_parent_pytest_context(tmp_path):
    parent_config = _walltime_config(tmp_path, "0:02:00", walltime_limit="0:01:00")
    dirsafe_runtag = parent_config.config__dirsafe_runtag
    parent_context = ExecutionContext(
        commit_hash="abc",
        config=parent_config,
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        command_line_args=[],
        run_descriptor=build_run_descriptor(
            stamp="parent",
            dirsafe_runtag=dirsafe_runtag,
            entrypoint="pytest",
            pid=1,
        ),
        is_debug_mode=True,
        is_no_build=True,
    )
    parent_context.save_self_to_out_file()

    submit_context = ExecutionContext(
        commit_hash="abc",
        config=_walltime_config(parent_context.unique_out_dir, "0:02:00", walltime_limit="0:01:00"),
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        command_line_args=[],
        run_descriptor=build_run_descriptor(
            stamp="child",
            dirsafe_runtag=dirsafe_runtag,
            entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
            pid=2,
        ),
        is_debug_mode=True,
        is_no_build=True,
    )
    submit_context.record_qsub_submission("0:01:00", "12345", submit_context.unique_out_dir)
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
    assert ExecutionContext.find_stamped_run_context(
        submit_context.unique_out_dir,
        parent_context.config.config__dirsafe_runtag,
        entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
        require_continuation=True,
    ).unique_out_dir == submit_context.unique_out_dir


def test_discover_run_contexts_finds_single_train_context(tmp_path):
    parent_config = _walltime_config(tmp_path, "0:01:00")
    dirsafe_runtag = parent_config.config__dirsafe_runtag
    parent_context = ExecutionContext(
        commit_hash="abc",
        config=parent_config,
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        command_line_args=[],
        run_descriptor=build_run_descriptor(
            stamp="parent",
            dirsafe_runtag=dirsafe_runtag,
            entrypoint="pytest",
            pid=1,
        ),
        is_debug_mode=True,
        is_no_build=True,
    )

    single_train_context = ExecutionContext(
        commit_hash="abc",
        config=_walltime_config(parent_context.unique_out_dir, "0:01:00"),
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        command_line_args=[],
        run_descriptor=build_run_descriptor(
            stamp="child",
            dirsafe_runtag=dirsafe_runtag,
            entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
            pid=2,
        ),
        is_debug_mode=True,
        is_no_build=True,
    )
    single_train_context.save_self_to_out_file()

    discovered_contexts = ExecutionContext.discover_run_contexts(
        parent_context.unique_out_dir,
        entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
        dirsafe_runtag=dirsafe_runtag,
    )

    assert [(context.unique_out_dir, path.parent) for context, path in discovered_contexts] == [
        (single_train_context.unique_out_dir, single_train_context.unique_out_dir)
    ]
