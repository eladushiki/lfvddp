"""Durable receipts around qsub for the canonical ATLAS routine."""

import json
import os
import re
from pathlib import Path
from atlas_routine.files import atomic_json

QUOTA_ERROR = re.compile(
    r"qsub:.*(?:per-user limit|would exceed.*(?:limit|quota))", re.I
)

RECEIPT_ENV = "LFVDDP_SUBMISSION_INTENT_RECEIPT"


class SubmissionReceipt:
    def __init__(self, path: Path, context, job_name: str):
        state = Path(os.environ["SUBMISSION_STATE_PATH"])
        expected = state.parent / "submission-state-tools/intents"
        if (
            path.resolve().parent != expected.resolve()
            or not state.with_suffix(".lock").is_dir()
        ):
            raise ValueError(
                "Submission receipt must belong to the held canonical transaction"
            )
        self.path = path
        self.data = json.loads(path.read_text())
        if not re.fullmatch(r"[a-f0-9]{32}", self.data["id"]):
            raise ValueError("Invalid submission intent token")
        if any(call["status"] != "accepted" for call in self.data["calls"]):
            raise RuntimeError("Unresolved qsub call: do not retry")
        self.call = {
            "status": "prepared",
            "pbs_name": "a" + self.data["id"][:10] + f"{len(self.data['calls']):02d}",
            "original_name": job_name,
            "kind": "train" if job_name.endswith("_exec_train") else "other",
            "context_directory": str(context.unique_out_dir),
            "walltime": context.config.cluster__qsub_walltime,
        }
        self.data["calls"].append(self.call)
        self.flush()

    @classmethod
    def from_environment(cls, context, job_name):
        filename = os.environ.get(RECEIPT_ENV)
        return cls(Path(filename), context, job_name) if filename else None

    def flush(self):
        atomic_json(self.path, self.data)

    def started(self):
        self.call["status"] = "started"
        self.flush()

    def accepted(self, job_id: str):
        if not re.fullmatch(r"\d+(?:\[\])?(?:\.[\w.-]+)?", job_id):
            raise ValueError("qsub did not return a scheduler job ID")
        self.call.update(status="accepted", job_id=job_id)
        self.flush()

    def failed(self, message: str, *, definite: bool):
        self.call.update(status="rejected" if definite else "uncertain", error=message)
        self.flush()
