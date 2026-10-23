"""Saved FIFO submission intents and conservative interrupted-qsub recovery."""

import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from atlas_routine.completion import completion_evidence, output_path
from atlas_routine.files import atomic_json
from atlas_routine.scheduler import parent
from atlas_routine.state import now
from atlas_routine.submission_intent import RECEIPT_ENV, QUOTA_ERROR

QUOTA = QUOTA_ERROR


class SubmissionManager:
    def __init__(self, store, scheduler, backend, report):
        self.store, self.scheduler, self.backend, self.report = (
            store,
            scheduler,
            backend,
            report,
        )

    def cleanup(self, entry, intent, receipt):
        if any(call.get("job_id") for call in receipt["calls"]):
            raise ValueError("Cannot clean a possibly submitted run")
        output = (self.store.root / entry["output_root"]).resolve()
        directories = set(output.glob("*_run_of_submit_train.py_*")) - {
            Path(p) for p in intent["previous_directories"]
        }
        for directory in sorted(directories):
            context = json.loads((directory / "context.json").read_text())
            if context.get("run_successful") is not False or context.get(
                "qsub_submissions"
            ):
                raise ValueError("Pre-qsub residue could not be verified")
            directory.resolve().relative_to(output)
            evidence = {
                "path": str(directory),
                "at": now(),
                "basis": "false context; empty qsub history; intent has no accepted scheduler call",
            }
            entry.setdefault("residue_cleanup", []).append(evidence)
            self.store.save()
            shutil.rmtree(directory)
            evidence["verified_removed_at"] = now()
            self.store.save()

    def finalize(self, entry, intent, receipt):
        calls = [call for call in receipt["calls"] if call["kind"] == "train"]
        if (
            not calls
            or any(call["status"] != "accepted" for call in calls)
            or any(
                call["status"] in ("started", "uncertain") for call in receipt["calls"]
            )
        ):
            raise RuntimeError("Incomplete submission intent requires reconciliation")
        directories = {call["context_directory"] for call in calls}
        if len(directories) != 1:
            raise ValueError("Intent returned multiple training directories")
        directory = Path(directories.pop()).resolve()
        (self.store.root / entry["output_root"]).resolve().relative_to(self.store.root)
        directory.relative_to((self.store.root / entry["output_root"]).resolve())
        ids = [call["job_id"].split(".")[0] for call in calls]
        snapshot = self.scheduler.refresh()
        if (
            not all(snapshot.live([job_id]) for job_id in ids)
            and not completion_evidence(directory, ids)["accepted"]
        ):
            raise RuntimeError(
                "Returned array absent from scheduler and verified outputs"
            )
        context_file = directory / "context.json"
        context = json.loads(context_file.read_text())
        if (
            intent["kind"] != "continuation"
            and context.get("commit_hash") != intent["commit"]
        ):
            raise ValueError("Submission context differs from intended training commit")
        # Receipt closes qsub->context-save crash window; reconstruct only recorded calls.
        for call in calls:
            if not any(
                parent(record["job_id"]) == parent(call["job_id"])
                for record in context.get("qsub_submissions", [])
            ):
                context.setdefault("qsub_submissions", []).append(
                    {
                        "chunk_index": len(context.get("qsub_submissions", [])) + 1,
                        "walltime": call["walltime"],
                        "job_id": call["job_id"].split(".")[0],
                        "submitted_at": now(),
                        "submit_run_dir": str(directory),
                    }
                )
                context["qsub_walltime_chunk"] = None
        atomic_json(context_file, context)
        attempt = {
            "kind": intent["kind"],
            "job_ids": ids,
            "submitted_at": now(),
            "remote_commit": intent["commit"],
            "branch": intent["branch"],
            "intent_id": intent["id"],
            "command": intent["command"],
            "scheduler_outcome": "active"
            if snapshot.live(ids)
            else "inactive_verified",
        }
        attempt["auxiliary_job_ids"] = [
            call["job_id"]
            for call in receipt["calls"]
            if call["kind"] != "train" and call.get("job_id")
        ]
        if intent["kind"] == "continuation":
            expected = context["config"]["cluster__qsub_walltime"]
            correction = self.backend.correct_walltime(entry, expected)
            attempt.update(
                extra_time=intent["extra_time"], source_config_update=correction
            )
            entry.pop("pending_continuation", None)
        entry.setdefault("attempts", []).append(attempt)
        entry.update(
            status="submitted",
            remote_commit=intent["commit"],
            remote_submission_directory=str(directory),
        )
        entry.setdefault("submission_intents", []).append(
            {**intent, "status": "verified", "verified_at": now(), "job_ids": ids}
        )
        entry.pop("pending_submission_intent", None)
        entry.pop("last_error", None)
        entry.pop("blocked_reason", None)
        self.store.save()
        self.report["submissions"].append(
            {"id": entry["id"], "kind": intent["kind"], "job_ids": ids}
        )

    def recover(self, entry):
        intent = entry["pending_submission_intent"]
        expected_receipt = (
            self.store.path.parent
            / "submission-state-tools/intents"
            / (intent["id"] + ".json")
        )
        if Path(intent["receipt_path"]).resolve() != expected_receipt.resolve():
            raise ValueError("Receipt is outside canonical intent storage")
        receipt = json.loads(expected_receipt.read_text())
        if receipt["id"] != intent["id"]:
            raise ValueError("Receipt identity differs from canonical intent")
        calls = receipt["calls"]
        not_started = not calls or all(call["status"] == "prepared" for call in calls)
        quota_rejected = bool(calls) and all(
            call["status"] == "rejected" and QUOTA.search(call.get("error", ""))
            for call in calls
        )
        if not_started or quota_rejected:
            self.cleanup(entry, intent, receipt)
            entry.pop("pending_submission_intent")
            entry["status"] = (
                "continuation_requested"
                if intent["kind"] == "continuation"
                else "requested"
            )
            self.store.save()
            return
        for call in calls:
            if call["status"] in ("started", "uncertain"):
                call.update(
                    job_id=self.scheduler.recover_call(call, intent["id"]),
                    status="accepted",
                )
            elif call["status"] == "rejected" and call["kind"] == "train":
                raise RuntimeError(
                    "Recorded qsub rejection requires failure handling before retry"
                )
        atomic_json(Path(intent["receipt_path"]), receipt)
        self.finalize(entry, intent, receipt)

    def submit(self, entry):
        snapshot = self.scheduler.refresh()
        branch, commit = self.backend.prepare(entry, self.store.data, snapshot)
        config, _ = self.backend.configs(entry)
        continuation = entry["status"] == "continuation_requested"
        if continuation:
            extra = entry["pending_continuation"]["extra_time"]
            args = [
                "--continue",
                str(output_path(self.store.root, entry)),
                "--extra-time",
                extra,
            ]
        else:
            args = ["--configs", *(entry.get("config_packs") or [entry["config_pack"]])]
            if entry.get("only_train"):
                args.append("--only-train")
            if entry.get("debug"):
                args.append("--debug")
            args += ["--out-dir", entry["output_root"]]
        token = uuid.uuid4().hex
        path = (
            self.store.path.parent / "submission-state-tools/intents" / f"{token}.json"
        )
        intent = {
            "id": token,
            "receipt_path": str(path),
            "kind": "continuation"
            if continuation
            else "rerun"
            if entry.get("attempts")
            else "initial",
            "created_at": now(),
            "branch": branch,
            "commit": commit,
            "command": ["python", "-m", "train.submit_train", *args],
            "previous_directories": [
                str(p)
                for p in (self.store.root / entry["output_root"]).glob(
                    "*_run_of_submit_train.py_*"
                )
            ],
        }
        if continuation:
            intent["extra_time"] = extra
        receipt = {"id": token, "request_id": entry["id"], "calls": []}
        atomic_json(path, receipt)
        entry["pending_submission_intent"] = intent
        self.store.save()  # No invocation of qsub may precede this checkpoint.
        environment = dict(os.environ, **{RECEIPT_ENV: str(path)})
        environment.pop("LFVDDP_COMMIT_HASH", None)
        label = "submit-" + token
        result = self.backend.command(
            [
                sys.executable,
                os.environ["LFVDDP_ATLAS_RUNNER"],
                "--worker",
                "submit",
                *args,
            ],
            label,
            env=environment,
        )
        receipt = json.loads(path.read_text())
        if any(call.get("job_id") for call in receipt["calls"]):
            self.recover(entry)
            return None
        log = (self.backend.runtime / (label + ".log")).read_text()
        if (
            result.returncode
            and QUOTA.search(log)
            and receipt["calls"]
            and all(call["status"] == "rejected" for call in receipt["calls"])
        ):
            self.cleanup(entry, intent, receipt)
            entry["last_error"] = {
                "stage": "quota",
                "at": now(),
                "log": str(self.backend.runtime / (label + ".log")),
            }
            entry.pop("pending_submission_intent")
            self.store.save()
            self.report["quota_rejected"].append(entry["id"])
            return config.cluster__qsub_n_jobs
        if not receipt["calls"] or all(
            call["status"] in ("prepared", "rejected") for call in receipt["calls"]
        ):
            self.cleanup(entry, intent, receipt)
            entry.pop("pending_submission_intent")
        raise RuntimeError(f"Submission failed or is uncertain; inspect {label}.log")
