"""One completion criterion for plotting, pruning and archival."""

import hashlib
import json
import re
from pathlib import Path
from frame.cluster.atlas.scheduler import parent

ACCEPTANCE_FRACTION = 0.9


def accepted_size(successful: int, expected: int) -> bool:
    return (
        type(expected) is int
        and expected > 0
        and successful > ACCEPTANCE_FRACTION * expected
    )


def frozen(entry: dict) -> bool:
    paths = str(
        [
            entry.get(k)
            for k in (
                "config_pack",
                "config_packs",
                "output_root",
                "remote_submission_directory",
                "remote_multi_run_directory",
            )
        ]
    )
    return bool(re.search(r"(?:plots-v|/v)[1-4](?:/|\b)", paths))


def protected(entry: dict) -> bool:
    return "research workflow" in entry.get("purpose", "").lower() or bool(
        entry.get("full_product_archive")
    )


def output_path(root: Path, entry: dict) -> Path:
    output = (root / entry["output_root"]).resolve()
    output.relative_to(root)
    submission = (root / entry["remote_submission_directory"]).resolve()
    submission.relative_to(output)
    if "_run_of_" not in submission.name:
        raise ValueError("Expected a saved timestamped submission directory")
    return submission


def completion_evidence(directory: Path, job_ids: list[str]) -> dict:
    context = json.loads((directory / "context.json").read_text())
    expected = context["config"]["cluster__qsub_n_jobs"]
    if type(expected) is not int or expected < 1:
        raise ValueError("Invalid expected array size")
    workers = {}
    invalid = []
    superseded = []
    for file in sorted(directory.glob("*_run_of_single_train.py_*/context.json")):
        try:
            value = json.loads(file.read_text())
            index = value.get("array_index")
            if index is None and expected == 1:
                index = 1
            if type(index) is not int or index not in range(1, expected + 1):
                invalid.append({"path": str(file), "error": "Invalid array index"})
                continue
            if value.get("run_successful") is True:
                if index in workers:
                    raise ValueError(f"Duplicate successful array element {index}")
                workers[index] = (file, value)
            else:
                superseded.append((file, value))
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            invalid.append({"path": str(file), "error": str(error)})
    for file, value in superseded:
        replacement = workers.get(value.get("array_index"))
        if (
            replacement
            and value.get("run_successful") is False
            and set(file.parent.iterdir()) == {file}
            and isinstance(value.get("time"), str)
            and value["time"] < replacement[1].get("time", "")
            and not value.get("products")
        ):
            continue
        invalid.append(
            {
                "path": str(file),
                "array_index": value.get("array_index"),
                "error": "Unsuccessful worker",
            }
        )
    parents = {parent(job_id) for job_id in job_ids}
    successful_logs = set()
    failures = []
    walltime = False
    for log in sorted(directory.glob("*.OU*")):
        match = re.fullmatch(r"(\d+)(?:\[(\d+)\])?(?:\.[^.]+)*\.OU[^/]*", log.name)
        if not match or match[1] not in parents:
            continue
        index = int(match[2]) if match[2] is not None else 1
        if match[2] is None and expected != 1:
            raise ValueError("Array output lacks an element index")
        if index not in range(1, expected + 1):
            raise ValueError("PBS log index outside expected array")
        raw = log.read_bytes()
        text = raw.decode(errors="replace").strip()
        if re.search(r"Job exit status:\s*0\s*$", text):
            if index in successful_logs:
                raise ValueError("Duplicate exit-zero PBS log")
            successful_logs.add(index)
        else:
            walltime |= "PBS: job killed: walltime" in text
            failures.append(
                {
                    "path": str(log),
                    "index": index,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "text": text,
                }
            )
    successful = sorted(set(workers) & successful_logs)
    missing = sorted(set(range(1, expected + 1)) - set(successful))
    return {
        "expected_elements": expected,
        "accepted_successful_elements": len(successful),
        "successful_indices": successful,
        "excluded_indices": missing,
        "failed_logs": failures,
        "invalid_contexts": invalid,
        "walltime_kill": walltime,
        "accepted": accepted_size(len(successful), expected),
        "complete": len(successful) == expected and not invalid,
        "scheduler_inactive": True,
    }
