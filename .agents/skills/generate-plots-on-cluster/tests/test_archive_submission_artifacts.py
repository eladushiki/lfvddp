"""Tests for the cluster artifact-archive helper's directory selection."""

import importlib.util
import json
import tarfile
from pathlib import Path

import pytest

from frame.context.run_descriptor import build_run_descriptor
from frame.file_structure import (
    CONFIGS_DIR_NAME,
    CONTEXT_FILE_NAME,
    CREATE_PLOTS_SCRIPT_NAME,
    SINGLE_TRAIN_SCRIPT_NAME,
    SUBMIT_TRAIN_SCRIPT_NAME,
)

_HELPER_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/archive_submission_artifacts.py"
)
_HELPER_SPEC = importlib.util.spec_from_file_location(
    "archive_submission_artifacts", _HELPER_PATH
)
assert _HELPER_SPEC is not None and _HELPER_SPEC.loader is not None
archive_submission_artifacts = importlib.util.module_from_spec(_HELPER_SPEC)
_HELPER_SPEC.loader.exec_module(archive_submission_artifacts)


def _run_directory(root: Path, entrypoint: str, pid: int) -> Path:
    directory = root / build_run_descriptor("stamp", "tag", entrypoint, pid)
    directory.mkdir()
    return directory


def test_helper_uses_project_run_descriptors_for_directory_selection(tmp_path):
    submission = _run_directory(tmp_path, SUBMIT_TRAIN_SCRIPT_NAME, 1)
    (submission / CONTEXT_FILE_NAME).write_text('{"is_debug_mode": true}')
    (submission / CONFIGS_DIR_NAME).mkdir()
    training = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, 2)
    plot = _run_directory(submission, CREATE_PLOTS_SCRIPT_NAME, 3)
    arbitrary_artifact = submission / "artifact.txt"
    arbitrary_artifact.touch()

    removable = archive_submission_artifacts.removable_children(submission)

    assert removable == [arbitrary_artifact, training]
    assert archive_submission_artifacts.retained_worker_source(removable) == training
    (submission / CONTEXT_FILE_NAME).write_text("{}")
    assert archive_submission_artifacts.retained_worker_source(removable) == training
    assert archive_submission_artifacts.all_submission_directories(tmp_path) == [
        submission
    ]
    assert plot not in removable


def test_cli_archives_submission_and_retains_one_worker(tmp_path, monkeypatch):
    submission = _run_directory(tmp_path, SUBMIT_TRAIN_SCRIPT_NAME, 1)
    (submission / CONTEXT_FILE_NAME).write_text("{}")
    (submission / CONFIGS_DIR_NAME).mkdir()
    retained_worker = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, 2)
    archived_worker = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, 3)
    artifact = submission / "artifact.txt"
    artifact.write_text("artifact")

    monkeypatch.setattr(
        "sys.argv",
        [
            str(_HELPER_PATH),
            "--results-root",
            str(tmp_path),
            str(submission),
        ],
    )

    assert archive_submission_artifacts.main() == 0
    assert (submission / archive_submission_artifacts.ARCHIVE_NAME).is_file()
    assert retained_worker.is_dir()
    assert not archived_worker.exists()
    assert not artifact.exists()


@pytest.mark.parametrize("failed_context", ['{"run_successful": false}', "{", None])
def test_archive_preserves_accepted_partial_array_evidence(tmp_path, failed_context):
    submission = _run_directory(tmp_path, SUBMIT_TRAIN_SCRIPT_NAME, 1)
    (submission / CONTEXT_FILE_NAME).write_text("{}")
    (submission / CONFIGS_DIR_NAME).mkdir()
    workers = [
        _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, pid)
        for pid in range(2, 14)
    ]
    for worker in workers[:-1]:
        (worker / CONTEXT_FILE_NAME).write_text(json.dumps({"run_successful": True}))
    failed = workers[-1]
    if failed_context is not None:
        (failed / CONTEXT_FILE_NAME).write_text(failed_context)
    history = failed / "training_outcomes"
    history.mkdir()
    (history / "history.json").write_text("{}")
    log = submission / "123[12].pbs.OU"
    log.write_text("Job exit status: 137\n")

    archive_submission_artifacts.archive_submission(
        submission, dry_run=True, temporary_directory=None
    )
    assert failed.is_dir()
    assert not (submission / archive_submission_artifacts.ARCHIVE_NAME).exists()
    archive_submission_artifacts.archive_submission(
        submission, dry_run=False, temporary_directory=None
    )

    with tarfile.open(
        submission / archive_submission_artifacts.ARCHIVE_NAME
    ) as archive:
        assert archive.extractfile(log.name).read() == b"Job exit status: 137\n"
        assert f"{failed.name}/training_outcomes/history.json" in archive.getnames()
        if failed_context is not None:
            assert (
                archive.extractfile(f"{failed.name}/context.json").read().decode()
                == failed_context
            )
    retained = min(workers)
    assert retained.is_dir()
    assert all(not worker.exists() for worker in workers if worker != retained)
    assert not log.exists()


@pytest.fixture
def continued_submission(tmp_path):
    submission = _run_directory(tmp_path, SUBMIT_TRAIN_SCRIPT_NAME, 1)
    (submission / CONFIGS_DIR_NAME).mkdir()
    (submission / CONTEXT_FILE_NAME).write_text(
        json.dumps(
            {
                "config": {"cluster__qsub_n_jobs": 2},
                "qsub_submissions": [{"job_id": "123[].pbs"}, {"job_id": "456[].pbs"}],
            }
        )
    )
    for index in (1, 2):
        worker = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, index + 1)
        (worker / CONTEXT_FILE_NAME).write_text('{"run_successful": true}')
        (worker / "training_outcomes").mkdir()
        (worker / "training_outcomes/history.h5").write_text("history")
        (worker / "runtime_resources.json").write_text("{}")
        (submission / f"456[{index}].pbs.OU").write_text("Job exit status: 0\n")
    (submission / "123[1].pbs.OU").write_text("PBS: job killed: walltime exceeded\n")
    (submission / "123[2].pbs.OU").write_text("Job exit status: 0\n")
    plot = _run_directory(submission, CREATE_PLOTS_SCRIPT_NAME, 4)
    (plot / "t_train_percentile_progression_plot.png").write_bytes(b"plot")
    return submission


@pytest.mark.parametrize("all_under_root", [False, True])
def test_cli_prunes_recovered_array_and_archives_failure_evidence(
    continued_submission, monkeypatch, all_under_root
):
    submission = continued_submission
    failure = submission / "123[1].pbs.OU"
    original = failure.read_bytes()
    args = [
        str(_HELPER_PATH),
        "--results-root",
        str(submission.parent),
        "--prune-finished-intermediates",
        str(submission),
    ]
    if all_under_root:
        args[-1] = "--all-under-root"
    monkeypatch.setattr("sys.argv", [*args, "--dry-run"])
    assert archive_submission_artifacts.main() == 0
    assert len(list(submission.glob("**/*.h5"))) == 2
    assert len(list(submission.glob("*.OU"))) == 4
    monkeypatch.setattr("sys.argv", args)
    assert archive_submission_artifacts.main() == 0
    assert not list(submission.glob("**/*.h5"))
    assert not list(submission.glob("**/training_outcomes"))
    assert not list(submission.glob("**/runtime_resources*.json"))
    assert list(submission.glob("*.OU")) == [failure]
    assert failure.read_bytes() == original
    assert len(list(submission.glob("*/context.json"))) == 2
    assert archive_submission_artifacts.has_progression_plot(submission)
    assert archive_submission_artifacts.main() == 0
    monkeypatch.setattr(
        "sys.argv", [a for a in args if a != "--prune-finished-intermediates"]
    )
    assert archive_submission_artifacts.main() == 0
    with tarfile.open(
        submission / archive_submission_artifacts.ARCHIVE_NAME
    ) as archive:
        assert archive.extractfile(failure.name).read() == original


@pytest.mark.parametrize(
    "case",
    [
        "missing_latest",
        "latest_walltime",
        "latest_error",
        "historical_error",
        "unknown_parent",
        "out_of_range",
        "duplicate_output",
        "missing_size",
        "duplicate_parent",
        "unrecorded_parent",
        "failed_worker",
        "missing_plot",
        "empty_logs",
        "missing_worker",
    ],
)
def test_pruning_refuses_incomplete_or_unverified_recovery(continued_submission, case):
    submission = continued_submission
    if case == "missing_latest":
        (submission / "456[2].pbs.OU").unlink()
    elif case in ("latest_walltime", "latest_error"):
        (submission / "456[2].pbs.OU").write_text(
            "PBS: job killed: walltime"
            if case == "latest_walltime"
            else "Job exit status: 1"
        )
    elif case == "historical_error":
        (submission / "123[1].pbs.OU").write_text("Job exit status: 1")
    elif case in ("unknown_parent", "out_of_range", "duplicate_output"):
        name = {
            "unknown_parent": "789[1].pbs.OU",
            "out_of_range": "456[3].pbs.OU",
            "duplicate_output": "456[1].pbs.OU.duplicate",
        }[case]
        (submission / name).write_text("Job exit status: 0")
    elif case in ("missing_size", "duplicate_parent", "unrecorded_parent"):
        path = submission / CONTEXT_FILE_NAME
        context = json.loads(path.read_text())
        if case == "missing_size":
            context["config"] = {}
        elif case == "duplicate_parent":
            context["qsub_submissions"][1]["job_id"] = "123[].pbs"
        else:
            context["qsub_submissions"] = [{"job_id": "123[].pbs"}]
        path.write_text(json.dumps(context))
    elif case == "failed_worker":
        next(submission.glob("*/context.json")).write_text('{"run_successful": false}')
    elif case == "missing_plot":
        next(submission.glob("*/*progression*.png")).unlink()
    elif case == "missing_worker":
        import shutil

        shutil.rmtree(next(submission.glob("*_run_of_single_train.py_*")))
    else:
        for log in submission.glob("*.OU*"):
            log.unlink()
    before = {p: p.read_bytes() for p in submission.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        archive_submission_artifacts.prune_finished_intermediates(
            submission, dry_run=False
        )
    assert {p: p.read_bytes() for p in submission.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("continued", [False, True])
def test_zero_exit_logs_still_prune_normally(continued_submission, continued):
    submission = continued_submission
    (submission / "123[1].pbs.OU").write_text("Job exit status: 0")
    if not continued:
        (submission / CONTEXT_FILE_NAME).write_text("{}")
    archive_submission_artifacts.prune_finished_intermediates(submission, dry_run=False)
    assert not list(submission.glob("*.OU*"))
    assert not list(submission.glob("**/*.h5"))


def test_multiple_recovered_walltime_chunks(continued_submission):
    submission = continued_submission
    path = submission / CONTEXT_FILE_NAME
    context = json.loads(path.read_text())
    context["qsub_submissions"].insert(1, {"job_id": "234[]"})
    path.write_text(json.dumps(context))
    failure = submission / "234[2].pbs.OU"
    failure.write_text("PBS: job killed: walltime exceeded")
    archive_submission_artifacts.prune_finished_intermediates(submission, dry_run=False)
    assert {p.name for p in submission.glob("*.OU")} == {"123[1].pbs.OU", failure.name}


@pytest.fixture
def superseded_launch_submission(continued_submission):
    submission = continued_submission
    workers = sorted(submission.glob("*_run_of_single_train.py_*"))
    for index, worker in enumerate(workers, 1):
        (worker / CONTEXT_FILE_NAME).write_text(
            json.dumps(
                {
                    "run_successful": True,
                    "array_index": index,
                    "time": "run_20261019_100000",
                    "products": ["final-statistics.json"],
                }
            )
        )
        (worker / "final-statistics.json").write_text('{"t": 1.0}')
    old = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, 5)
    (old / CONTEXT_FILE_NAME).write_text(
        json.dumps(
            {
                "run_successful": False,
                "array_index": 1,
                "time": "run_20261019_095900",
                "products": [],
            }
        )
    )
    return submission


@pytest.mark.parametrize("all_under_root", [False, True])
@pytest.mark.parametrize("continued", [False, True])
def test_cli_prunes_superseded_launches_without_removing_contexts(
    superseded_launch_submission, monkeypatch, all_under_root, continued
):
    submission = superseded_launch_submission
    if not continued:
        path = submission / CONTEXT_FILE_NAME
        context = json.loads(path.read_text())
        context["qsub_submissions"] = context["qsub_submissions"][-1:]
        path.write_text(json.dumps(context))
        for log in submission.glob("123*.OU"):
            log.unlink()
    preserved = {
        p: p.read_bytes()
        for p in submission.rglob("*")
        if p.is_file()
        and (
            p.name in (CONTEXT_FILE_NAME, "final-statistics.json") or p.suffix == ".png"
        )
    }
    args = [
        str(_HELPER_PATH),
        "--results-root",
        str(submission.parent),
        "--prune-finished-intermediates",
        "--all-under-root" if all_under_root else str(submission),
    ]
    monkeypatch.setattr("sys.argv", [*args, "--dry-run"])
    assert archive_submission_artifacts.main() == 0
    assert len(list(submission.glob("**/*.h5"))) == 2
    monkeypatch.setattr("sys.argv", args)
    assert archive_submission_artifacts.main() == 0
    assert not list(submission.glob("**/*.h5"))
    assert not list(submission.glob("**/training_outcomes"))
    assert not list(submission.glob("**/runtime_resources*.json"))
    assert {p: p.read_bytes() for p in preserved} == preserved
    expected_logs = [submission / "123[1].pbs.OU"] if continued else []
    assert list(submission.glob("*.OU")) == expected_logs
    assert archive_submission_artifacts.main() == 0


@pytest.mark.parametrize(
    "case",
    [
        "artifact",
        "empty_directory",
        "products",
        "unknown_index",
        "missing_index",
        "boolean_index",
        "later_launch",
        "equal_time",
        "missing_time",
        "malformed_context",
        "missing_success_flag",
        "missing_successful_worker",
        "duplicate_success",
        "out_of_range_success",
        "missing_size",
        "missing_log",
        "duplicate_log",
        "unknown_log",
    ],
)
def test_superseded_launch_pruning_refuses_ambiguous_or_incomplete_evidence(
    superseded_launch_submission, case
):
    submission = superseded_launch_submission
    old = next(
        p
        for p in submission.glob("*_run_of_single_train.py_*")
        if json.loads((p / CONTEXT_FILE_NAME).read_text())["run_successful"] is False
    )
    path = old / CONTEXT_FILE_NAME
    context = json.loads(path.read_text())
    if case == "artifact":
        (old / "partial.h5").write_text("partial result")
    elif case == "empty_directory":
        (old / "training_outcomes").mkdir()
    elif case == "products":
        context["products"] = ["missing-but-recorded-result"]
    elif case == "unknown_index":
        context["array_index"] = 3
    elif case == "missing_index":
        del context["array_index"]
    elif case == "boolean_index":
        context["array_index"] = True
    elif case == "later_launch":
        context["time"] = "run_20261019_110000"
    elif case == "equal_time":
        context["time"] = "run_20261019_100000"
    elif case == "missing_time":
        del context["time"]
    elif case == "missing_success_flag":
        del context["run_successful"]
    elif case == "missing_successful_worker":
        worker = next(
            p
            for p in submission.glob("*_run_of_single_train.py_*")
            if json.loads((p / CONTEXT_FILE_NAME).read_text())["run_successful"] is True
        )
        (worker / CONTEXT_FILE_NAME).write_text('{"run_successful": false}')
    elif case in ("duplicate_success", "out_of_range_success"):
        worker = next(
            p
            for p in submission.glob("*_run_of_single_train.py_*")
            if json.loads((p / CONTEXT_FILE_NAME).read_text())["array_index"] == 2
        )
        data = json.loads((worker / CONTEXT_FILE_NAME).read_text())
        data["array_index"] = 1 if case == "duplicate_success" else 3
        (worker / CONTEXT_FILE_NAME).write_text(json.dumps(data))
    elif case == "missing_size":
        parent = submission / CONTEXT_FILE_NAME
        data = json.loads(parent.read_text())
        data["config"] = {}
        parent.write_text(json.dumps(data))
    elif case == "missing_log":
        (submission / "456[2].pbs.OU").unlink()
    elif case in ("duplicate_log", "unknown_log"):
        (
            submission
            / (
                "456[1].pbs.OU.duplicate"
                if case == "duplicate_log"
                else "789[1].pbs.OU"
            )
        ).write_text("Job exit status: 0")
    path.write_text("{" if case == "malformed_context" else json.dumps(context))
    before = {p: p.read_bytes() for p in submission.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        archive_submission_artifacts.prune_finished_intermediates(
            submission, dry_run=False
        )
    assert {p: p.read_bytes() for p in submission.rglob("*") if p.is_file()} == before
