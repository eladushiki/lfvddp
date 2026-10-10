#!/usr/bin/env python3
"""Run a cluster state transaction under a shared, fail-fast directory lock."""

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 2 * 60 * 60
TERMINATION_GRACE_SECONDS = 10


def group_alive(group: int) -> bool:
    """Ignore zombies, which cannot write or perform workflow actions."""
    rows = subprocess.check_output(["ps", "-eo", "pgid=,stat="], text=True, timeout=5)
    return any(
        int(fields[0]) == group and not fields[1].startswith("Z")
        for row in rows.splitlines()
        if len(fields := row.split()) == 2
    )


def stop_group(child: subprocess.Popen, grace: float) -> None:
    """Stop all child-group writers before releasing their shared lock."""
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace
        while True:
            child.poll()  # Reap the direct child before checking the group.
            if not group_alive(child.pid):
                child.wait()
                return
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
    raise RuntimeError("Child group still alive; retaining submission-state lock")


def run_locked(
    project_root: Path,
    command: list[str],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    termination_grace_seconds: float = TERMINATION_GRACE_SECONDS,
) -> int:
    """Hold the canonical lock until the command has exited."""
    if not all(
        math.isfinite(value) and value > 0
        for value in (timeout_seconds, termination_grace_seconds)
    ):
        raise ValueError("Lock timeout and termination grace must be positive")
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
    child = None
    release = True
    interrupted = None
    previous_handlers = {}

    def interrupted_by(signum, _frame):
        nonlocal interrupted
        interrupted = signum

    try:
        owner.write_text(
            json.dumps(
                {
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                    "acquired_at": datetime.now(UTC).isoformat(),
                    "command": command,
                    "timeout_seconds": timeout_seconds,
                }
            )
            + "\n"
        )
        environment = os.environ.copy()
        environment["SUBMISSION_STATE_PATH"] = str(state)
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted_by)
        child = subprocess.Popen(
            command, cwd=root, env=environment, start_new_session=True
        )
        deadline = time.monotonic() + timeout_seconds
        while child.poll() is None:
            if interrupted is not None:
                return 128 + interrupted
            if time.monotonic() >= deadline:
                print(
                    "Submission-state transaction timed out; stopping child group",
                    file=sys.stderr,
                )
                return 124
            time.sleep(0.1)
        return child.returncode
    finally:
        try:
            if child is not None:
                stop_group(child, termination_grace_seconds)
        except BaseException:
            release = False
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            if release:
                owner.unlink(missing_ok=True)
                lock.rmdir()


def main() -> int:
    root = os.environ.get("WIS_CLUSTER_REMOTE_PROJECT_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not root or not command:
        print(
            "Use the configured cluster SSH shell, then run: python <helper> -- <command> [args...]",
            file=sys.stderr,
        )
        return 2
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return run_locked(Path(root), command, args.timeout_seconds)


if __name__ == "__main__":
    sys.exit(main())
