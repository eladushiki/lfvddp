"""Tests for the cluster artifact-archive helper's directory selection."""

import importlib.util
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
