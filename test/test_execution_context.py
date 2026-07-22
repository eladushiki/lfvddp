from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from data_tools.data_utils import DataSet
from frame.command_line.handle_args import parse_config_from_args
from frame.context.execution_context import (
    ExecutionContext,
    create_config_from_paramters,
    version_controlled_execution_context,
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
from frame.file_system.textual_data import (
    load_config_file,
    load_dict_from_json,
    save_dict_to_json,
)
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


def _mixed_generated_and_resampled_config(tmp_path, numpy_path: Path):
    config_params = {}
    for config_path in DEFAULT_CONFIG_PATHS.values():
        config_params.update(load_config_file(config_path))
    config_params.update(load_config_file(
        Path("test/configs/detector/basic_1D_detector_config.json")
    ))
    config_params.update(load_config_file(
        Path("test/configs/train/short_1D_train_config_without_nuisance.json")
    ))

    generated_definitions = load_config_file(
        Path("test/configs/dataset/disjoint_1D_generated_dataset_config.json")
    )["dataset__definitions"]
    loaded_definitions = {
        category: {
            "name": category.split("_")[0].upper(),
            "type": "loaded",
            "category": category,
            "dataset_loaded__file_name": str(numpy_path),
            "dataset_loaded__observable_naming": {"param_0": "param_0"},
            "dataset_loaded__resample_is_resample": True,
            "dataset_loaded__resample_is_replacement": False,
            "dataset__number_of_background_events": 8,
        }
        for category in ("a_cr", "b_cr")
    }
    config_params["dataset__definitions"] = [
        loaded_definitions.get(definition["category"], definition)
        for definition in generated_definitions
    ]
    return create_config_from_paramters(
        config_params,
        out_dir=str(tmp_path),
        plot_in_place=True,
    )


def _context_args(continue_from=None) -> Namespace:
    if continue_from is not None:
        return Namespace(
            continue_from=continue_from,
        )
    return Namespace(
        debug=True,
        no_build=True,
        only_train=False,
        continue_from=None,
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


def test_continuation_prepares_next_chunk_before_yield(tmp_path):
    config = _walltime_config(
        tmp_path,
        "0:02:00",
        walltime_limit="0:01:00",
    )
    context = ExecutionContext(
        commit_hash="abc",
        config=config,
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        command_line_args=["submit_train.py", "--configs"],
        run_descriptor=build_run_descriptor(
            stamp="submit",
            dirsafe_runtag=config.config__dirsafe_runtag,
            entrypoint=SUBMIT_TRAIN_SCRIPT_NAME,
            pid=1,
        ),
        is_debug_mode=True,
        is_no_build=True,
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


def test_continuation_recreates_generated_and_resampled_datasets(
    tmp_path,
    data_batch_events,
):
    numpy_path = tmp_path / "events.npy"
    np.save(numpy_path, np.arange(40, dtype=float).reshape(-1, 1))
    initial_seed_path = tmp_path / "initial_seed.json"
    save_dict_to_json({"random_seed": 314159}, initial_seed_path)
    initial_config_paths = [*DEFAULT_CONFIG_PATHS.values(), initial_seed_path]

    initial_config = _mixed_generated_and_resampled_config(tmp_path, numpy_path)
    with version_controlled_execution_context(
        config=initial_config,
        config_paths=initial_config_paths,
        command_line_args=["pytest"],
        args=_context_args(),
    ) as initial_context:
        initial_events = data_batch_events(initial_context)
        initial_run_dir = initial_context.unique_out_dir
        for category in (
            DataSet.DataSetCategory.A_CR,
            DataSet.DataSetCategory.B_CR,
        ):
            assert initial_events[category].shape == (8, 1)
            assert not np.array_equal(
                initial_events[category],
                np.arange(8, dtype=float).reshape(-1, 1),
            )

    save_dict_to_json({"random_seed": 271828}, initial_seed_path)
    with version_controlled_execution_context(
        config=None,
        config_paths=None,
        command_line_args=["pytest", "--continue", str(initial_run_dir)],
        args=_context_args(continue_from=initial_run_dir),
    ) as continuation_context:
        continuation_events = data_batch_events(continuation_context)

        assert continuation_context.unique_out_dir == initial_run_dir
        assert continuation_context.random_seed == 314159
        assert continuation_context.config_paths == initial_config_paths
        assert continuation_context.command_line_args == ["pytest"]
        assert initial_events.keys() == continuation_events.keys()
        for category in initial_events:
            np.testing.assert_array_equal(
                continuation_events[category],
                initial_events[category],
            )


def test_child_context_loading_matches_array_index(tmp_path):
    config = _walltime_config(tmp_path, "0:01:00")
    contexts = []
    for array_index, random_seed in ((1, 101), (2, 202)):
        context = ExecutionContext(
            commit_hash="abc",
            config=_walltime_config(tmp_path, "0:01:00"),
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
            is_no_build=True,
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


def test_continuation_is_the_only_accepted_option(capsys):
    config_paths, args = parse_config_from_args(["--continue", "results/run"])

    assert config_paths is None
    assert args.continue_from == Path("results/run")

    with pytest.raises(SystemExit):
        parse_config_from_args(["--continue", "results/run", "--debug"])
    capsys.readouterr()
