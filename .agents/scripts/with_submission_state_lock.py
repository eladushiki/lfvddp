#!/usr/bin/env python3
"""Run a cluster state transaction under a shared, fail-fast directory lock."""

import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def run_locked(project_root: Path, command: list[str]) -> int:
    """Hold the canonical lock until the command has exited."""
    root = project_root.resolve(strict=True)
    state = root / ".agents" / "submission-state.yaml"
    lock = state.with_suffix(".lock")
    try:
        lock.mkdir()
    except FileExistsError:
        print(
            f"Submission state is locked: {lock}. Do not remove an active lock.",
            file=sys.stderr,
        )
        return 75
    owner = lock / "owner.json"
    try:
        owner.write_text(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "acquired_at": datetime.now(UTC).isoformat(),
                    "command": command,
                }
            )
            + "\n"
        )
        environment = os.environ.copy()
        environment["SUBMISSION_STATE_PATH"] = str(state)
        return subprocess.run(
            command, cwd=root, env=environment, check=False
        ).returncode
    finally:
        owner.unlink(missing_ok=True)
        lock.rmdir()


def main() -> int:
    root = os.environ.get("WIS_CLUSTER_REMOTE_PROJECT_ROOT")
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not root or not command:
        print(
            "Use the configured cluster SSH shell, then run: python <helper> -- <command> [args...]",
            file=sys.stderr,
        )
        return 2
    return run_locked(Path(root), command)


if __name__ == "__main__":
    sys.exit(main())
