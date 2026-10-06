"""Tests for the cluster artifact-archive helper's directory selection."""

import importlib.util
import shutil
import tarfile
from pathlib import Path

from frame.context.run_descriptor import build_run_descriptor
from frame.file_structure import (
    CONFIGS_DIR_NAME,
    CONTEXT_FILE_NAME,
    CREATE_PLOTS_SCRIPT_NAME,
    SINGLE_TRAIN_SCRIPT_NAME,
    SUBMIT_TRAIN_SCRIPT_NAME,
)

_HELPER_PATH = Path(
    ".agents/skills/generate-plots-on-cluster/scripts/archive_submission_artifacts.py"
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
    assert (
        archive_submission_artifacts.debug_helper_source(submission, removable)
        == training
    )
    (submission / CONTEXT_FILE_NAME).write_text("{}")
    assert (
        archive_submission_artifacts.debug_helper_source(submission, removable) is None
    )
    assert archive_submission_artifacts.all_submission_directories(tmp_path) == [
        submission
    ]
    assert plot not in removable


def test_cli_archives_debug_submission_and_retains_one_worker(tmp_path, monkeypatch):
    submission = _run_directory(tmp_path, SUBMIT_TRAIN_SCRIPT_NAME, 1)
    (submission / CONTEXT_FILE_NAME).write_text('{"is_debug_mode": true}')
    (submission / CONFIGS_DIR_NAME).mkdir()
    retained_worker = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, 2)
    (retained_worker / "dataset_process_plot.png").write_bytes(b"visible plot")
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
    assert not (
        submission / archive_submission_artifacts.PREDICTION_PLOTS_DIR_NAME
    ).exists()


def test_archive_retains_one_workers_prediction_plots(tmp_path):
    submission = _run_directory(tmp_path, SUBMIT_TRAIN_SCRIPT_NAME, 1)
    (submission / CONTEXT_FILE_NAME).write_text("{}")
    (submission / CONFIGS_DIR_NAME).mkdir()
    first_worker = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, 2)
    second_worker = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, 3)
    first_plots = first_worker / "plots"
    first_plots.mkdir()
    (first_plots / "dataset_process_plot_a.png").write_bytes(b"first a")
    (first_plots / "dataset_process_plot_b.png").write_bytes(b"first b")
    (second_worker / "dataset_process_plot.png").write_bytes(b"second")

    archive_submission_artifacts.archive_submission(
        submission, dry_run=False, temporary_directory=None
    )

    retained = submission / archive_submission_artifacts.PREDICTION_PLOTS_DIR_NAME
    assert (retained / "plots/dataset_process_plot_a.png").read_bytes() == b"first a"
    assert (retained / "plots/dataset_process_plot_b.png").read_bytes() == b"first b"
    assert not first_worker.exists()
    assert not second_worker.exists()
    with tarfile.open(submission / archive_submission_artifacts.ARCHIVE_NAME) as tar:
        assert tar.getmember(f"{second_worker.name}/dataset_process_plot.png")


def test_archive_recovers_prediction_plots_from_existing_archive(tmp_path):
    submission = _run_directory(tmp_path, SUBMIT_TRAIN_SCRIPT_NAME, 1)
    (submission / CONTEXT_FILE_NAME).write_text("{}")
    (submission / CONFIGS_DIR_NAME).mkdir()
    worker = _run_directory(submission, SINGLE_TRAIN_SCRIPT_NAME, 2)
    (worker / "dataset_process_plot.png").write_bytes(b"archived plot")
    archive_submission_artifacts.archive_submission(
        submission, dry_run=False, temporary_directory=None
    )
    retained = submission / archive_submission_artifacts.PREDICTION_PLOTS_DIR_NAME
    shutil.rmtree(retained)

    archive_submission_artifacts.archive_submission(
        submission, dry_run=True, temporary_directory=None
    )
    assert not retained.exists()
    archive_submission_artifacts.archive_submission(
        submission, dry_run=False, temporary_directory=None
    )

    assert (retained / "dataset_process_plot.png").read_bytes() == b"archived plot"
