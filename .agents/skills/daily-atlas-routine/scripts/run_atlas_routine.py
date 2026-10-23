#!/usr/bin/env python3
"""Run the versioned ATLAS routine once; never open or retry SSH here."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def bootstrap():
    root = Path(os.environ.get("WIS_CLUSTER_REMOTE_PROJECT_ROOT", Path.cwd())).resolve()
    manifest = Path(__file__).parent / "manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        for name, expected in metadata["sha256"].items():
            file = (manifest.parent / name).resolve()
            file.relative_to(manifest.parent.resolve())
            if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    "Deployed runner checksum differs from committed package"
                )
        os.environ["LFVDDP_ATLAS_REVISION"] = metadata["revision"]
    sys.path.insert(0, str(root))
    os.environ["LFVDDP_ATLAS_RUNNER"] = str(Path(__file__).resolve())
    return root


def main():
    root = bootstrap()
    if sys.argv[1:2] == ["--worker"]:
        from atlas_routine.state import assert_held_lock

        assert_held_lock(root)
        kind, args = sys.argv[2], sys.argv[3:]
        if kind == "submit":
            from atlas_routine.submit_entrypoint import main as worker
        elif kind == "plot":
            from atlas_routine.plot_entrypoint import main as worker
        else:
            raise ValueError("Unknown routine worker")
        worker(args)
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--record-prepared-commit", help="Record GitHub App-prepared idle main checkout"
    )
    parser.add_argument("--phase", choices=("all", "plots", "submit"), default="all")
    parser.add_argument(
        "--held-lock",
        action="store_true",
        help="Reuse the existing canonical transaction",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect without workflow or state changes",
    )
    parser.add_argument(
        "--archive-completed-state",
        action="store_true",
        help="Explicitly archive terminal history and clear active lists, only when idle",
    )
    args = parser.parse_args()
    if args.dry_run and (args.archive_completed_state or args.record_prepared_commit):
        parser.error(
            "Use a plain dry run before explicit reset or preparation commands"
        )
    if not args.held_lock:
        health = subprocess.run([sys.executable, "-c", "import torch"], cwd=root)
        if health.returncode:
            for command in (["quota"], ["lquota"]):
                subprocess.run(command, check=False)
            return health.returncode
        lock = Path(__file__).with_name("with_submission_state_lock.py")
        # Fail-fast helper is the sole lock implementation; no nested acquisition.
        return subprocess.call(
            [
                sys.executable,
                str(lock),
                "--",
                sys.executable,
                str(Path(__file__).resolve()),
                "--held-lock",
                *sys.argv[1:],
            ],
            cwd=root,
        )
    from atlas_routine.runner import Runner
    from atlas_routine.state import StateStore

    runner = Runner(StateStore(root))
    report = runner.run(
        archive_completed=args.archive_completed_state,
        dry_run=args.dry_run,
        phase=args.phase,
        prepared_commit=args.record_prepared_commit,
    )
    print(json.dumps(report, default=str), flush=True)
    return 2 if report.get("errors") or report.get("requires_review") else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(
            json.dumps({"error": type(error).__name__, "message": str(error)}),
            file=sys.stderr,
        )
        sys.exit(2)
