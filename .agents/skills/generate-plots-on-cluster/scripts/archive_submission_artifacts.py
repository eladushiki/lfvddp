#!/usr/bin/env python3
"""Archive removable array-job artifacts from verified plot submissions."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path


RESULTS_ROOT = Path("results/highlights/2026-09").resolve()
ARCHIVE_NAME = "array-job-artifacts.tar.gz"


def checked_submission(path_arg: str) -> Path:
    submission = Path(path_arg).resolve()
    try:
        submission.relative_to(RESULTS_ROOT)
    except ValueError as error:
        raise ValueError(f"submission is outside {RESULTS_ROOT}: {submission}") from error
    if not submission.is_dir():
        raise ValueError(f"submission is not a directory: {submission}")
    if not (submission / "context.json").is_file():
        raise ValueError(f"submission has no context.json: {submission}")
    if not (submission / "configs").is_dir():
        raise ValueError(f"submission has no configs directory: {submission}")
    return submission


def is_pbs_log(path: Path) -> bool:
    name = path.name
    return path.is_file() and (
        ".pbs.o" in name or ".pbs.e" in name or name.endswith(".o") or name.endswith(".e")
    )


def removable_children(submission: Path) -> list[Path]:
    return sorted(
        child
        for child in submission.iterdir()
        if "_run_of_single_train.py_" in child.name or is_pbs_log(child)
    )


def archive_submission(submission: Path, *, dry_run: bool) -> None:
    archive = submission / ARCHIVE_NAME
    sources = removable_children(submission)
    if archive.exists() and sources:
        raise ValueError(
            f"archive already exists while removable artifacts remain: {archive}; "
            "inspect it before retrying"
        )
    if not sources:
        print(f"unchanged: {submission}")
        return

    print(f"archive: {archive}")
    for source in sources:
        print(f"  {source.name}")
    if dry_run:
        return

    temporary_archive = archive.with_suffix(archive.suffix + ".tmp")
    try:
        with tarfile.open(temporary_archive, "w:gz") as tar:
            for source in sources:
                tar.add(source, arcname=source.name, recursive=True)
        with tarfile.open(temporary_archive, "r:gz") as tar:
            archived = set(tar.getnames())
        missing = [source.name for source in sources if source.name not in archived]
        if missing:
            raise ValueError(f"archive verification missing: {', '.join(missing)}")
        temporary_archive.replace(archive)
        for source in sources:
            if source.is_dir():
                shutil.rmtree(source)
            else:
                source.unlink()
    except Exception:
        temporary_archive.unlink(missing_ok=True)
        raise
    print(f"archived and removed {len(sources)} items: {submission}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="+", help="tracked submission directories")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        for path_arg in args.submission:
            archive_submission(checked_submission(path_arg), dry_run=args.dry_run)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
