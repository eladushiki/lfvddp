"""Bounded PBS observations, reused until an explicit refresh is necessary."""

import json
import os
import re
import subprocess
from collections import Counter
from dataclasses import dataclass

INACTIVE = {"X", "F", "C"}


def parent(job_id: str) -> str:
    return re.split(r"[\[.]", job_id)[0]


@dataclass
class Snapshot:
    active_rows: list[list[str]]
    history_rows: list[list[str]]

    @property
    def counts(self) -> dict:
        return dict(Counter(row[-2] for row in self.active_rows))

    @property
    def idle(self) -> bool:
        return all(row[-2] in INACTIVE for row in self.active_rows)

    def rows_for(self, job_ids: list[str], *, history: bool = False) -> list[list[str]]:
        parents = {parent(job_id) for job_id in job_ids}
        return [
            row
            for row in (self.history_rows if history else self.active_rows)
            if parent(row[0]) in parents
        ]

    def live(self, job_ids: list[str]) -> bool:
        return any(row[-2] not in INACTIVE for row in self.rows_for(job_ids))

    def observed(self, job_ids: list[str]) -> bool:
        rows = self.active_rows + self.history_rows
        return all(
            any(parent(row[0]) == parent(job_id) for row in rows) for job_id in job_ids
        )


class Scheduler:
    def __init__(self, command=subprocess.check_output):
        self.command = command
        self.snapshot = None
        self.queries = 0

    def output(self, args: list[str]) -> str:
        self.queries += 1
        return self.command(["qstat", *args], text=True, timeout=40)

    def refresh(self) -> Snapshot:
        rows = []
        for flag in ("-tu", "-xu"):
            text = self.output([flag, os.environ["USER"]])
            rows.append(
                [line.split() for line in text.splitlines() if re.match(r"^\d", line)]
            )
        self.snapshot = Snapshot(*rows)
        return self.snapshot

    def recover_call(self, call: dict, token: str) -> str:
        jobs = json.loads(
            self.output(["-x", "-f", "-F", "json", "-u", os.environ["USER"]])
        ).get("Jobs", {})
        matches = set()
        for job_id, job in jobs.items():
            variables = job.get("Variable_List", {})
            if isinstance(variables, str):
                variables = dict(
                    part.split("=", 1) for part in variables.split(",") if "=" in part
                )
            if (
                job.get("Job_Name") == call["pbs_name"]
                and job.get("Job_Owner", "").split("@")[0] == os.environ["USER"]
                and variables.get("LFVDDP_SUBMISSION_INTENT") == token
            ):
                matches.add((parent(job_id), "[]" if "[" in job_id else ""))
        if len(matches) != 1:
            raise RuntimeError(
                "Submission outcome is uncertain; refusing to repeat qsub"
            )
        number, suffix = matches.pop()
        return number + suffix
