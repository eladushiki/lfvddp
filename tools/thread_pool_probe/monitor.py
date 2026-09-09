"""Sample all Linux tasks in the probe job's session without importing ML code."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import time
from typing import Optional


def _stat_fields(path: Path) -> Optional[tuple[str, list[str]]]:
    try:
        contents = path.read_text()
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    closing_parenthesis = contents.rfind(")")
    if closing_parenthesis < 0:
        return None
    comm = contents[contents.find("(") + 1 : closing_parenthesis]
    return comm, contents[closing_parenthesis + 2 :].split()


def _process_session(process_dir: Path) -> Optional[int]:
    parsed = _stat_fields(process_dir / "stat")
    if parsed is None:
        return None
    _, fields = parsed
    try:
        return int(fields[3])
    except (IndexError, ValueError):
        return None


def sample_session(
    session_id: int,
    *,
    excluded_pid: Optional[int] = None,
    proc_root: Path = Path("/proc"),
) -> dict:
    """Return one task-state snapshot for processes in ``session_id``."""

    processes = []
    state_totals: Counter[str] = Counter()
    for process_dir in proc_root.iterdir():
        if not process_dir.name.isdigit():
            continue
        pid = int(process_dir.name)
        if pid == excluded_pid or _process_session(process_dir) != session_id:
            continue

        states: Counter[str] = Counter()
        thread_names: Counter[str] = Counter()
        try:
            task_dirs = tuple((process_dir / "task").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        for task_dir in task_dirs:
            parsed = _stat_fields(task_dir / "stat")
            if parsed is None:
                continue
            thread_name, fields = parsed
            if fields:
                states[fields[0]] += 1
                thread_names[thread_name] += 1

        if not states:
            continue
        state_totals.update(states)
        parsed = _stat_fields(process_dir / "stat")
        comm = parsed[0] if parsed is not None else "unknown"
        processes.append(
            {
                "pid": pid,
                "comm": comm,
                "threads": sum(states.values()),
                "runnable": states["R"],
                "states": dict(sorted(states.items())),
                "thread_names": dict(sorted(thread_names.items())),
            }
        )

    processes.sort(key=lambda process: process["pid"])
    total = sum(state_totals.values())
    runnable = state_totals["R"]
    sleeping = sum(state_totals[state] for state in ("D", "I", "S"))
    return {
        "threads": total,
        "runnable": runnable,
        "sleeping": sleeping,
        "other": total - runnable - sleeping,
        "states": dict(sorted(state_totals.items())),
        "processes": processes,
    }


def _peak_record(sample: dict) -> dict:
    return {
        "threads": sample["threads"],
        "runnable": sample["runnable"],
        "sleeping": sample["sleeping"],
        "other": sample["other"],
        "states": sample["states"],
        "processes": sample["processes"],
    }


def monitor(case: str, interval: float) -> None:
    """Sample until SIGTERM and print one compact machine-readable summary."""

    should_stop = False

    def request_stop(_signum, _frame) -> None:
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    own_pid = os.getpid()
    session_id = os.getsid(0)
    sample_count = 0
    peak_total: Optional[dict] = None
    peak_runnable: Optional[dict] = None

    while not should_stop:
        sample = sample_session(session_id, excluded_pid=own_pid)
        sample_count += 1
        if peak_total is None or sample["threads"] > peak_total["threads"]:
            peak_total = _peak_record(sample)
        if peak_runnable is None or sample["runnable"] > peak_runnable["runnable"]:
            peak_runnable = _peak_record(sample)
        time.sleep(interval)

    summary = {
        "case": case,
        "excluded_monitor_pid": own_pid,
        "sample_interval_seconds": interval,
        "samples": sample_count,
        "peak_total": peak_total,
        "peak_runnable": peak_runnable,
    }
    print("THREAD_PROBE_RESULT=" + json.dumps(summary, separators=(",", ":")), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--interval", type=float, default=0.1)
    arguments = parser.parse_args()
    if arguments.interval <= 0:
        parser.error("--interval must be positive")
    monitor(arguments.case, arguments.interval)


if __name__ == "__main__":
    main()
