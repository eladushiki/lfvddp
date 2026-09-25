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


ARCHIVE_NAME = "array-job-artifacts.tar.gz"


def checked_submission(path_arg: str, results_root: Path) -> Path:
    submission = Path(path_arg).resolve()
    try:
        submission.relative_to(results_root)
    except ValueError as error:
        raise ValueError(f"submission is outside {results_root}: {submission}") from error
    if not submission.is_dir():
        raise ValueError(f"submission is not a directory: {submission}")
    if not (submission / "context.json").is_file():
        raise ValueError(f"submission has no context.json: {submission}")
    if not (submission / "configs").is_dir():
        raise ValueError(f"submission has no configs directory: {submission}")
    return submission


def removable_children(submission: Path) -> list[Path]:
    retained = {"context.json", "configs", ARCHIVE_NAME}
    return sorted(
        child
        for child in submission.iterdir()
        if child.name not in retained and "_run_of_create_plots.py_" not in child.name
    )


def debug_helper_source(submission: Path, sources: list[Path]) -> Path | None:
    """Keep one array-worker directory in place for debug submissions."""
    context = json.loads((submission / "context.json").read_text())
    if context.get("is_debug_mode") is not True:
        return None
    return next(
        (
            source
            for source in sources
            if source.is_dir() and "_run_of_single_train.py_" in source.name
        ),
        None,
    )


def training_outcome_directories(sources: list[Path]) -> list[Path]:
    """Return training histories that must survive archive verification."""
    return [
        path
        for source in sources
        if source.is_dir()
        for path in source.glob("**/training_outcomes")
        if path.is_dir()
    ]


def all_submission_directories(results_root: Path) -> list[Path]:
    if not results_root.is_dir():
        raise ValueError(f"results root does not exist: {results_root}")
    return sorted(
        path
        for path in results_root.glob("**/run_*_run_of_submit_train.py_pid_*")
        if path.is_dir()
    )


def archive_submission(
    submission: Path, *, dry_run: bool, temporary_directory: Path | None
) -> None:
    archive = submission / ARCHIVE_NAME
    sources = removable_children(submission)
    debug_helper = debug_helper_source(submission, sources)
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
                        tar.addfile(member, previous.extractfile(member) if member.isfile() else None)
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.all_under_root == bool(args.submission):
        parser.error("supply submission directories or --all-under-root, but not both")
    try:
        results_root = args.results_root.resolve()
        paths = all_submission_directories(results_root) if args.all_under_root else args.submission
        temporary_directory = (
            args.temporary_directory.resolve() if args.temporary_directory else None
        )
        if temporary_directory is not None and not temporary_directory.is_dir():
            raise ValueError(f"temporary directory is not a directory: {temporary_directory}")
        for path_arg in paths:
            submission = checked_submission(str(path_arg), results_root)
            if args.restore:
                restore_submission(submission, dry_run=args.dry_run)
            else:
                archive_submission(
                    submission,
                    dry_run=args.dry_run,
                    temporary_directory=temporary_directory,
                )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
