"""Existing LFVDDP entry points behind one timed command boundary."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from atlas_routine.artifacts import Artifacts
from atlas_routine.files import atomic_json
from atlas_routine.state import now


class PreparationRequired(RuntimeError):
    """An unsatisfied saved checkout gate; keep request priority unchanged."""


class Backend:
    def __init__(self, root: Path, runtime: Path, report: dict):
        self.root, self.runtime, self.report = root, runtime, report
        self.artifacts = Artifacts(root)

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
        from frame.context.execution_context import create_config_from_paramters
        from frame.file_system.textual_data import (
            expand_config_paths,
            load_config_params_from_paths,
        )

        packs = entry.get("config_packs") or [entry["config_pack"]]
        files = expand_config_paths([self.root / pack for pack in packs])
        parameters = load_config_params_from_paths(files)
        config = create_config_from_paramters(parameters, out_dir=entry["output_root"])
        return config, files

    def validate_comparison(self, members: list[dict], *, saved_only=False):
        """Check planned packs against actual saved runs before any group side effect."""
        from frame.context.execution_context import ExecutionContext
        from train.comparison_config import (
            saved_comparison_configs,
            validate_comparison_configs,
        )

        configs = []
        for entry in members:
            directory = entry.get("remote_submission_directory")
            if directory and (
                saved_only or entry["status"] not in ("requested", "blocked")
            ):
                configs.extend(saved_comparison_configs(self.root / directory))
            elif saved_only:
                raise ValueError(f"Missing saved run configuration: {entry['id']}")
            else:
                config, _ = self.configs(entry)
                configs.append((entry["id"], ExecutionContext.serialize(config)))
        validate_comparison_configs(configs)

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
        if self.git("--no-optional-locks", "status", "--porcelain=v1"):
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

    def correct_walltime(self, entry: dict, expected: str) -> dict:
        _, files = self.configs(entry)
        import yaml

        from atlas_routine.files import atomic_bytes

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
