"""Existing LFVDDP entry points behind one timed command boundary."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from frame.cluster.atlas.files import atomic_json
from frame.cluster.atlas.state import now


class PreparationRequired(RuntimeError):
    """An unsatisfied saved checkout gate; keep request priority unchanged."""


class Backend:
    def __init__(self, root: Path, runtime: Path, report: dict):
        self.root, self.runtime, self.report = root, runtime, report
        self.helper = None

    def command(
        self, args: list[str], label: str, *, env=None
    ) -> subprocess.CompletedProcess:
        self.runtime.mkdir(parents=True, exist_ok=True)
        log = self.runtime / f"{label}.log"
        started = time.monotonic()
        with log.open("w") as output:
            result = subprocess.run(
                args, cwd=self.root, stdout=output, stderr=subprocess.STDOUT, env=env
            )
        self.report["timings"].append(
            {
                "stage": label,
                "seconds": time.monotonic() - started,
                "exit_code": result.returncode,
                "log": str(log),
            }
        )
        return result

    def configs(self, entry: dict):
        from frame.file_system.textual_data import (
            expand_config_paths,
            load_config_params_from_paths,
        )
        from frame.context.execution_context import create_config_from_paramters

        packs = entry.get("config_packs") or [entry["config_pack"]]
        if "config_packs" in entry:
            match = re_match_pack(packs)
            version, dimension, source = match
            if packs[:2] != [
                f"configs/{version}/generic",
                f"configs/{version}/mandatory-optional/{dimension}/{source}",
            ]:
                raise ValueError("Saved configuration layer order is invalid")
        files = expand_config_paths([self.root / pack for pack in packs])
        parameters = load_config_params_from_paths(files)
        config = create_config_from_paramters(parameters, out_dir=entry["output_root"])
        if any("plots-v8/" in pack for pack in packs):
            if (
                config.train__epochs,
                config.train__optimizer_strategy,
                config.plot__prediction_process_number_of_bins,
            ) != (40000, "adam_lbfgs", 25):
                raise ValueError("Saved v8 runtime requirements are not satisfied")
            corrected = any(
                attempt.get("source_config_update", {}).get("after")
                == config.cluster__qsub_walltime
                for attempt in entry.get("attempts", [])
            )
            if (
                entry["status"] != "continuation_requested"
                and config.cluster__qsub_walltime != "12:00:00"
                and not corrected
            ):
                raise ValueError(
                    "v8 walltime differs without an audited continuation correction"
                )
            if "plot__prediction_process_share_observable_range" in parameters:
                raise ValueError("Removed observable-range key is present")
        return config, files

    def git(self, *args) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=self.root, text=True, timeout=40
        ).strip()

    def prepare(self, entry: dict, state: dict, snapshot) -> tuple[str, str]:
        branch = self.git("branch", "--show-current")
        required = entry.get("required_submission_branch") or branch
        for migration in state.get("branch_migrations", []):
            if (
                entry["status"] in migration["applies_to_statuses"]
                and required == migration["from"]
            ):
                required = migration["to"]
        if self.git("status", "--porcelain"):
            raise PreparationRequired("Checkout is dirty")
        if branch != required:
            if not snapshot.idle:
                raise PreparationRequired(
                    "Required branch differs while scheduler jobs are active"
                )
            raise PreparationRequired(
                f"Prepare required branch {required} with the repository GitHub App before retrying"
            )
        commit = self.git("rev-parse", "HEAD")
        action = entry.get("required_pre_submission_action")
        verification = entry.get("pre_submission_action_verification", {})
        owner = json.loads(
            (self.root / ".agents/submission-state.lock/owner.json").read_text()
        )
        if action and (
            verification.get("remote_commit") != commit
            or (
                snapshot.idle
                and verification.get("lock_acquired_at") != owner["acquired_at"]
            )
        ):
            raise PreparationRequired(
                "Saved pre-submission action needs verified GitHub App preparation"
            )
        if any("plots-v8/" in pack for pack in entry.get("config_packs", [])):
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", "20516a2b", "HEAD"],
                cwd=self.root,
                check=True,
            )
        return branch, commit

    def plot(
        self, entry: dict, directory: Path, *, background=None, members=None
    ) -> list[str]:
        pending = entry["pending_plot"]
        previous = set(pending["previous_directories"])

        def completed():
            outputs = []
            for child in directory.glob("*_run_of_create_plots.py_*"):
                if str(child) in previous:
                    continue
                context = child / "context.json"
                if (
                    context.exists()
                    and json.loads(context.read_text()).get("run_successful") is True
                ):
                    outputs.extend(str(p) for p in child.glob("*.png"))
            return outputs

        plots = completed()
        if not plots:
            if members is None:
                args = [sys.executable, "-m", "plot.create_plots", str(directory)]
            else:
                plan = self.runtime / f"{entry['id']}-plot-members.json"
                atomic_json(
                    plan,
                    {
                        "directory": str(directory),
                        "background": str(background),
                        "members": [str(p) for p in members],
                    },
                )
                args = [
                    sys.executable,
                    os.environ["LFVDDP_ATLAS_RUNNER"],
                    "--worker",
                    "plot",
                    str(plan),
                ]
            if entry.get("debug"):
                args.append("--debug")
            result = self.command(args, "plot-" + entry["id"])
            if result.returncode:
                raise RuntimeError("Plot entrypoint failed; see stage log")
            plots = completed()
        if not plots or (
            members is None
            and not any("t_train_percentile_progression_plot" in p for p in plots)
        ):
            raise ValueError("Expected configured plot products are missing")
        for path in plots:
            with open(path, "rb") as image:
                if image.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("Invalid PNG output")
        return plots

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
                    shutil.rmtree(source) if source.is_dir() else source.unlink()
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

    def correct_walltime(self, entry: dict, expected: str) -> dict:
        _, files = self.configs(entry)
        import yaml
        from frame.cluster.atlas.files import atomic_bytes

        for path in reversed(files):
            raw = path.read_text()
            value = json.loads(raw) if path.suffix == ".json" else yaml.safe_load(raw)
            if "cluster__qsub_walltime" in value:
                before = value["cluster__qsub_walltime"]
                value["cluster__qsub_walltime"] = expected
                content = (
                    json.dumps(value, indent=2)
                    if path.suffix == ".json"
                    else yaml.safe_dump(value, sort_keys=False)
                )
                atomic_bytes(path, content.encode())
                return {
                    "path": str(path),
                    "before": before,
                    "after": expected,
                    "updated_at": now(),
                }
        raise ValueError("No saved configuration layer defines walltime")


def re_match_pack(packs):
    import re

    if len(packs) != 3:
        raise ValueError("Expected three ordered configuration packs")
    match = re.match(r"configs/(plots-v\d+)/", packs[-1])
    if not match:
        raise ValueError("Invalid versioned pack")
    return (
        match[1],
        "4d" if "4d" in packs[-1] else "2d" if "2d" in packs[-1] else "1d",
        "cms_open_data" if "cms" in packs[-1] else "generated",
    )
