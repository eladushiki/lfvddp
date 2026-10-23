"""Verified artifact restoration, pruning and archive publication."""

import hashlib
import importlib.util
import shutil
import tarfile
from pathlib import Path
from atlas_routine.state import now


class Artifacts:
    def __init__(self, root: Path):
        self.root = root
        self.helper = None

    def archive_helper(self):
        if self.helper is None:
            file = (
                self.root
                / ".agents/skills/generate-plots-on-cluster/scripts/archive_submission_artifacts.py"
            )
            spec = importlib.util.spec_from_file_location("atlas_archive_helper", file)
            self.helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.helper)
        return self.helper

    def prune(self, directory: Path):
        helper = self.archive_helper()
        helper.prune_finished_intermediates(directory, dry_run=True)
        helper.prune_finished_intermediates(directory, dry_run=False)

    def restore(self, directory: Path):
        archive = directory / "array-job-artifacts.tar.gz"
        if not archive.exists():
            raise ValueError("Recorded archive is missing")
        with tarfile.open(archive) as tar:
            missing = []
            for member in tar.getmembers():
                destination = (directory / member.name).resolve()
                destination.relative_to(directory.resolve())
                if (
                    not (member.isdir() or member.isfile())
                    or Path(member.name).is_absolute()
                    or ".." in Path(member.name).parts
                ):
                    raise ValueError("Unsafe archive member")
                if member.isfile() and destination.exists():
                    if (
                        destination.is_symlink()
                        or hashlib.sha256(destination.read_bytes()).digest()
                        != hashlib.sha256(tar.extractfile(member).read()).digest()
                    ):
                        raise ValueError("Restored artifact conflicts with archive")
                else:
                    missing.append(member)
            tar.extractall(directory, members=missing, filter="data")
            for member in missing:
                if member.isfile():
                    if (
                        hashlib.sha256((directory / member.name).read_bytes()).digest()
                        != hashlib.sha256(tar.extractfile(member).read()).digest()
                    ):
                        raise ValueError("Restored artifact failed verification")

    def archive(self, directory: Path) -> dict:
        helper = self.archive_helper()
        archive = directory / helper.ARCHIVE_NAME
        sources = helper.removable_children(directory)
        retained = helper.retained_worker_source(sources)
        if archive.exists():
            # Recover interruption after archive verification, before source removal.
            with tarfile.open(archive) as tar:
                for source in sources:
                    if source == retained:
                        continue
                    files = source.rglob("*") if source.is_dir() else [source]
                    for file in files:
                        if file.is_file():
                            member = str(file.relative_to(directory))
                            try:
                                saved = tar.extractfile(member)
                            except KeyError as error:
                                raise ValueError(
                                    "Existing archive lacks remaining source"
                                ) from error
                            if (
                                hashlib.sha256(saved.read()).digest()
                                != hashlib.sha256(file.read_bytes()).digest()
                            ):
                                raise ValueError(
                                    "Existing archive differs from remaining source"
                                )
            for source in sources:
                if source != retained:
                    if source.is_dir():
                        shutil.rmtree(source)
                    else:
                        source.unlink()
        else:
            helper.archive_submission(directory, dry_run=True, temporary_directory=None)
            helper.archive_submission(
                directory, dry_run=False, temporary_directory=None
            )
        if not archive.is_file() or retained is None or not retained.is_dir():
            raise ValueError("Archive or retained worker missing")
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            if any(
                source.name not in names for source in sources if source != retained
            ):
                raise ValueError("Archive is missing source roots")
        return {
            "archive_path": str(archive),
            "completed_at": now(),
            "retained_worker_directories": [str(retained)],
        }
