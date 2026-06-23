from frame.context.execution_context import ExecutionContext, create_config_from_paramters
from frame.file_structure import CONTEXT_FILE_NAME
from frame.file_system.textual_data import load_config_file, load_dict_from_json
from test.environment import DEFAULT_CONFIG_PATHS


def _long_walltime_config(tmp_path):
    config_params = {}
    for config_path in DEFAULT_CONFIG_PATHS.values():
        config_params.update(load_config_file(config_path))
    config_params["cluster__qsub_walltime"] = "73:00:00"

    return create_config_from_paramters(
        config_params,
        out_dir=str(tmp_path),
        plot_in_place=True,
    )


def test_execution_context_persists_qsub_submission_state(tmp_path):
    context = ExecutionContext(
        commit_hash="abc",
        config=_long_walltime_config(tmp_path),
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
