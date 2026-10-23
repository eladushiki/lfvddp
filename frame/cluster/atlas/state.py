"""Canonical locked state and explicitly authorized completed-history reset."""

import copy
import gzip
import hashlib
import json
import os
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import yaml
from frame.cluster.atlas.files import atomic_bytes

LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
TERMINAL = {"analyzed", "retired"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_state(value: dict) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("version"), int):
        raise ValueError("State must have a version")
    for collection in ("submissions", "plot_groups"):
        entries = value[collection]
        identifiers = [entry["id"] for entry in entries]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate IDs in {collection}")
    identifiers = {entry["id"] for entry in value["submissions"]}
    for group in value["plot_groups"]:
        if (
            group["status"] != "retired"
            and not {group["background_submission"], *group["signal_submissions"]}
            <= identifiers
        ):
            raise ValueError(f"Missing members for {group['id']}")


def assert_held_lock(root: Path) -> Path:
    root = root.resolve(strict=True)
    state = root / ".agents/submission-state.yaml"
    if Path(os.environ.get("SUBMISSION_STATE_PATH", "")).resolve() != state:
        raise ValueError("Use the canonical lock helper before accessing state")
    owner = json.loads(state.with_suffix(".lock").joinpath("owner.json").read_text())
    if owner["host"] != socket.gethostname():
        raise ValueError("Lock belongs to another host")
    ancestor = os.getpid()
    while ancestor > 1 and ancestor != owner["pid"]:
        ancestor = int(
            subprocess.check_output(
                ["ps", "-o", "ppid=", "-p", str(ancestor)], text=True
            )
        )
    if ancestor != owner["pid"]:
        raise ValueError("Current process is not a child of the lock owner")
    return state


class StateStore:
    def __init__(self, root: Path):
        started = time.monotonic()
        self.root = root.resolve()
        self.path = assert_held_lock(self.root)
        self.raw = self.path.read_bytes()
        self.data = yaml.load(self.raw, Loader=LOADER)
        validate_state(self.data)
        self.timings = {
            "state_load_seconds": time.monotonic() - started,
            "state_save_seconds": 0.0,
        }
        self.saves = 0

    def save(self) -> None:
        started = time.monotonic()
        validate_state(self.data)
        content = yaml.dump(self.data, Dumper=DUMPER, sort_keys=False).encode()
        if yaml.load(content, Loader=LOADER) != self.data:
            raise ValueError("Serialized state failed round-trip validation")
        if self.path.read_bytes() != self.raw:
            raise RuntimeError("Canonical state changed outside this transaction")
        if content != self.raw:
            atomic_bytes(self.path, content)
            self.raw = content
            self.saves += 1
        self.timings["state_save_seconds"] += time.monotonic() - started

    def archive_completed(self, scheduler_idle: bool) -> dict:
        if not scheduler_idle:
            raise ValueError("State reset requires an inactive scheduler")
        if any(
            x["status"] not in TERMINAL or x.get("pending_submission_intent")
            for x in self.data["submissions"]
        ):
            raise ValueError(
                "State reset requires no unfinished submissions or intents"
            )
        if any(g["status"] not in TERMINAL for g in self.data["plot_groups"]):
            raise ValueError("State reset requires no unfinished plot groups")
        if not self.data["submissions"] and not self.data["plot_groups"]:
            return {"unchanged": True}
        digest = hashlib.sha256(self.raw).hexdigest()
        snapshot = (
            self.root / ".agents/submission-state-tools/history" / f"{digest}.yaml.gz"
        )
        if snapshot.exists():
            if gzip.decompress(snapshot.read_bytes()) != self.raw:
                raise ValueError("Existing history snapshot differs")
        else:
            atomic_bytes(snapshot, gzip.compress(self.raw, mtime=0))
        if hashlib.sha256(gzip.decompress(snapshot.read_bytes())).hexdigest() != digest:
            raise ValueError("History checksum verification failed")
        evidence = {
            "path": str(snapshot),
            "sha256": digest,
            "uncompressed_bytes": len(self.raw),
            "submissions": len(self.data["submissions"]),
            "plot_groups": len(self.data["plot_groups"]),
            "archived_at": now(),
        }
        replacement = copy.deepcopy(self.data)
        replacement.update(submissions=[], plot_groups=[], last_routine_summary={})
        replacement.setdefault("state_history", []).append(evidence)
        self.data = replacement
        self.save()
        return evidence
