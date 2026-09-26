#!/usr/bin/env python3
"""Archive removable array-job artifacts from verified plot submissions."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from frame.file_structure import (
    CONFIGS_DIR_NAME,
    CONTEXT_FILE_NAME,
    CREATE_PLOTS_SCRIPT_NAME,
    SINGLE_TRAIN_SCRIPT_NAME,
    SUBMIT_TRAIN_SCRIPT_NAME,
    TRAINING_OUTCOMES_DIR_NAME,
)
from frame.context.run_descriptor import parse_run_descriptor


ARCHIVE_NAME = "array-job-artifacts.tar.gz"
PROGRESSION_PLOT_NAME = "t_train_percentile_progression_plot"


def checked_submission(path_arg: str, results_root: Path) -> Path:
    submission = Path(path_arg).resolve()
    try:
        submission.relative_to(results_root)
    except ValueError as error:
        raise ValueError(
            f"submission is outside {results_root}: {submission}"
        ) from error
    if not submission.is_dir():
        raise ValueError(f"submission is not a directory: {submission}")
    if not (submission / CONTEXT_FILE_NAME).is_file():
        raise ValueError(f"submission has no {CONTEXT_FILE_NAME}: {submission}")
    if not (submission / CONFIGS_DIR_NAME).is_dir():
        raise ValueError(
            f"submission has no {CONFIGS_DIR_NAME} directory: {submission}"
        )
    return submission


def is_run_directory_for_entrypoint(directory: Path, entrypoint: str) -> bool:
    """Whether a generated run directory belongs to an entrypoint."""
    descriptor = parse_run_descriptor(directory.name)
    return descriptor is not None and descriptor.entrypoint == entrypoint


def removable_children(submission: Path) -> list[Path]:
    retained = {CONTEXT_FILE_NAME, CONFIGS_DIR_NAME, ARCHIVE_NAME}
    return sorted(
        child
        for child in submission.iterdir()
        if child.name not in retained
        and not is_run_directory_for_entrypoint(child, CREATE_PLOTS_SCRIPT_NAME)
    )


def debug_helper_source(
    sources: list[Path], *, retain_debug_helper: bool
) -> Path | None:
    """Keep one array-worker directory in place for debug submissions."""
    if not retain_debug_helper:
        return None
    return next(
        (
            source
            for source in sources
            if source.is_dir()
            and is_run_directory_for_entrypoint(source, SINGLE_TRAIN_SCRIPT_NAME)
        ),
        None,
    )


def training_outcome_directories(sources: list[Path]) -> list[Path]:
    """Return training histories that must survive archive verification."""
    return [
        path
        for source in sources
        if source.is_dir()
        for path in source.glob(f"**/{TRAINING_OUTCOMES_DIR_NAME}")
        if path.is_dir()
    ]


def has_progression_plot(submission: Path) -> bool:
    """Whether the single-submission history plot was persisted."""
    return any(
        path.is_file() and PROGRESSION_PLOT_NAME in path.name
        for path in submission.glob("**/*")
    )


def successful_worker_directories(submission: Path) -> list[Path]:
    """Validate every retained worker context before destructive pruning."""
    workers = sorted(
        directory
        for directory in submission.iterdir()
        if directory.is_dir()
        and is_run_directory_for_entrypoint(directory, SINGLE_TRAIN_SCRIPT_NAME)
    )
    if not workers:
        raise ValueError(f"submission has no worker directories: {submission}")
    failed = []
    for worker in workers:
        context_path = worker / CONTEXT_FILE_NAME
        try:
            successful = (
                json.loads(context_path.read_text()).get("run_successful") is True
            )
        except (OSError, json.JSONDecodeError):
            successful = False
        if not successful:
            failed.append(worker.name)
    if failed:
        raise ValueError(f"worker contexts are not successful: {', '.join(failed[:5])}")
    return workers


def verified_pbs_logs(submission: Path) -> list[Path]:
    """Return only PBS outputs that prove zero exit status for every worker."""
    logs = sorted(path for path in submission.glob("*.OU*") if path.is_file())
    if not logs:
        raise ValueError(f"submission has no PBS output logs: {submission}")
    invalid = [
        log.name
        for log in logs
        if not log.read_text(errors="ignore").rstrip().endswith("Job exit status: 0")
    ]
    if invalid:
        raise ValueError(f"PBS logs lack exit status 0: {', '.join(invalid[:5])}")
    return logs


def prune_finished_intermediates(submission: Path, *, dry_run: bool) -> None:
    """Remove regenerable worker artifacts after preserving progression plots."""
    h5_files = sorted(path for path in submission.glob("**/*.h5") if path.is_file())
    histories = sorted(
        path
        for path in submission.glob(f"**/{TRAINING_OUTCOMES_DIR_NAME}")
        if path.is_dir()
    )
    runtime_reports = sorted(
        path for path in submission.glob("**/runtime_resources*.json") if path.is_file()
    )
    if not (h5_files or histories or runtime_reports):
        print(f"unchanged: {submission}")
        return
    if not has_progression_plot(submission):
        raise ValueError(
            f"submission has no {PROGRESSION_PLOT_NAME} output: {submission}"
        )
    successful_worker_directories(submission)
    logs = verified_pbs_logs(submission)
    targets = [*h5_files, *histories, *runtime_reports, *logs]
    print(f"prune intermediates: {submission}")
    for target in targets:
        print(f"  {target.relative_to(submission)}")
    if dry_run:
        return
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    print(f"pruned {len(targets)} intermediate artifacts: {submission}")


def all_submission_directories(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        raise ValueError(f"results root does not exist: {results_root}")
    return sorted(
        context_path.parent
        for context_path in results_root.rglob(CONTEXT_FILE_NAME)
        if is_run_directory_for_entrypoint(
            context_path.parent, SUBMIT_TRAIN_SCRIPT_NAME
        )
    )


def archive_submission(
    submission: Path,
    *,
    dry_run: bool,
    temporary_directory: Path | None,
    retain_debug_helper: bool,
) -> None:
    archive = submission / ARCHIVE_NAME
    sources = removable_children(submission)
    debug_helper = debug_helper_source(sources, retain_debug_helper=retain_debug_helper)
    archive_sources = [source for source in sources if source != debug_helper]
    training_outcomes = training_outcome_directories(archive_sources)
    if not archive_sources:
        print(f"unchanged: {submission}")
        if debug_helper is not None:
            print(f"retained debug helper: {debug_helper.name}")
        return

    print(f"archive: {archive}")
    for source in archive_sources:
        print(f"  {source.name}")
    if debug_helper is not None:
        print(f"retain debug helper: {debug_helper.name}")
    if dry_run:
        return

    temporary_archive = (
        archive.with_suffix(archive.suffix + ".tmp")
        if temporary_directory is None
        else temporary_directory / f"{submission.name}.{uuid.uuid4().hex}.tar.gz.tmp"
    )
    sources_removed = False
    try:
        with tarfile.open(temporary_archive, "w:gz") as tar:
            if archive.is_file():
                with tarfile.open(archive, "r:gz") as previous:
                    for member in previous:
                        tar.addfile(
                            member,
                            previous.extractfile(member) if member.isfile() else None,
                        )
            for source in archive_sources:
                tar.add(source, arcname=source.name, recursive=True)
        with tarfile.open(temporary_archive, "r:gz") as tar:
            archived = set(tar.getnames())
        required_members = [
            *[source.name for source in archive_sources],
            *[str(path.relative_to(submission)) for path in training_outcomes],
        ]
        missing = [member for member in required_members if member not in archived]
        if missing:
            raise ValueError(f"archive verification missing: {', '.join(missing)}")
        if temporary_directory is None:
            temporary_archive.replace(archive)
        for source in archive_sources:
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
        sources_removed = True
        if temporary_directory is not None:
            shutil.move(str(temporary_archive), str(archive))
    except Exception:
        if not sources_removed:
            temporary_archive.unlink(missing_ok=True)
        raise
    print(f"archived and removed {len(archive_sources)} items: {submission}")
    if debug_helper is not None:
        print(f"retained debug helper: {debug_helper.name}")


def restore_submission(submission: Path, *, dry_run: bool) -> None:
    """Restore one archive without overwriting existing plot products."""
    archive = submission / ARCHIVE_NAME
    if not archive.is_file():
        raise ValueError(f"submission has no {ARCHIVE_NAME}: {submission}")

    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        destinations = []
        for member in members:
            destination = (submission / member.name).resolve()
            try:
                destination.relative_to(submission)
            except ValueError as error:
                raise ValueError(
                    f"archive contains an unsafe member {member.name!r}: {archive}"
                ) from error
            if member.isfile() and destination.exists():
                raise ValueError(
                    f"refusing to overwrite restored artifact {destination}: {archive}"
                )
            destinations.append(destination)

        print(f"restore: {archive}")
        if dry_run:
            for destination in destinations:
                print(f"  {destination.relative_to(submission)}")
            return
        tar.extractall(submission, members=members, filter="data")
    print(f"restored {len(members)} archived members: {submission}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="*", help="tracked submission directories")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="root containing the submission directories to archive",
    )
    parser.add_argument(
        "--all-under-root",
        action="store_true",
        help="archive every submission directory below --results-root",
    )
    parser.add_argument(
        "--temporary-directory",
        type=Path,
        help="build the verified archive here before removing sources; use only when the results filesystem has no temporary space",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="restore archived artifacts for aggregate plotting without deleting the archive",
    )
    parser.add_argument(
        "--retain-debug-helper",
        action="store_true",
        help="retain the first array-worker directory for debug inspection",
    )
    parser.add_argument(
        "--prune-finished-intermediates",
        action="store_true",
        help=(
            "remove verified-successful worker HDF5, training histories, runtime "
            "reports, and PBS logs after the percentile-progression plot exists"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.all_under_root == bool(args.submission):
        parser.error("supply submission directories or --all-under-root, but not both")
    if args.restore and args.prune_finished_intermediates:
        parser.error(
            "--restore and --prune-finished-intermediates are mutually exclusive"
        )
    try:
        results_root = args.results_root.resolve()
        paths = (
            all_submission_directories(results_root)
            if args.all_under_root
            else args.submission
        )
        temporary_directory = (
            args.temporary_directory.resolve() if args.temporary_directory else None
        )
        if temporary_directory is not None and not temporary_directory.is_dir():
            raise ValueError(
                f"temporary directory is not a directory: {temporary_directory}"
            )
        for path_arg in paths:
            submission = checked_submission(str(path_arg), results_root)
            if args.restore:
                restore_submission(submission, dry_run=args.dry_run)
            elif args.prune_finished_intermediates:
                prune_finished_intermediates(submission, dry_run=args.dry_run)
            else:
                archive_submission(
                    submission,
                    dry_run=args.dry_run,
                    temporary_directory=temporary_directory,
                    retain_debug_helper=args.retain_debug_helper,
                )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
