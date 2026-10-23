"""One foreground, checkpointed ATLAS transaction."""

import json
import os
import shutil
import time
from collections import Counter
from frame.cluster.atlas.backend import Backend, PreparationRequired
from frame.cluster.atlas.completion import (
    completion_evidence,
    frozen,
    output_path,
    protected,
)
from frame.cluster.atlas.scheduler import Scheduler
from frame.cluster.atlas.state import now
from frame.cluster.atlas.submission import SubmissionManager


class Runner:
    def __init__(self, store, scheduler=None, backend=None):
        self.store = store
        self.scheduler = scheduler or Scheduler()
        self.report = {
            "started_at": now(),
            "implementation_revision": os.environ.get(
                "LFVDDP_ATLAS_REVISION", "checkout"
            ),
            "events": [],
            "timings": [],
            "errors": [],
            "submissions": [],
            "quota_rejected": [],
            "quota_skipped": [],
            "deferred": [],
            "requires_review": [],
        }
        runtime = (
            store.path.parent / "submission-state-tools/runs" / now().replace(":", "-")
        )
        self.backend = backend or Backend(store.root, runtime, self.report)
        self.by_id = {x["id"]: x for x in store.data["submissions"]}

    def event(self, stage, entry):
        event = {"stage": stage, "id": entry["id"], "at": now()}
        self.report["events"].append(event)
        print(json.dumps(event), flush=True)

    def failure(self, entry, stage, error, *, block=False):
        entry["last_error"] = {"stage": stage, "at": now(), "message": str(error)}
        if block:
            entry.update(status="blocked", blocked_reason=str(error))
        self.store.save()
        self.report["errors"].append({"id": entry["id"], **entry["last_error"]})

    def reconcile(self):
        snapshot = self.scheduler.snapshot
        for entry in self.store.data["submissions"]:
            if entry["status"] == "retired" or frozen(entry):
                continue
            for attempt in entry.get("attempts", []):
                ids = attempt["job_ids"]
                attempt["last_scheduler_check"] = {
                    "observed_at": now(),
                    "element_states": dict(
                        Counter(r[-2] for r in snapshot.rows_for(ids))
                    ),
                    "history_states": dict(
                        Counter(r[-2] for r in snapshot.rows_for(ids, history=True))
                    ),
                }
            if entry["status"] != "submitted" or any(
                snapshot.live(a["job_ids"] + a.get("auxiliary_job_ids", []))
                for a in entry["attempts"]
            ):
                continue
            try:
                directory = output_path(self.store.root, entry)
                evidence = completion_evidence(
                    directory, entry["attempts"][-1]["job_ids"]
                )
                entry["completion_evidence"] = evidence
                if evidence["walltime_kill"]:
                    config = json.loads((directory / "context.json").read_text())[
                        "config"
                    ]
                    entry.update(
                        status="continuation_requested",
                        pending_continuation={
                            "extra_time": config["cluster__qsub_walltime"],
                            "evidence": evidence,
                        },
                    )
                elif evidence["accepted"]:
                    entry.update(
                        status="finished",
                        finished_at=now(),
                        accepted_successful_elements=evidence[
                            "accepted_successful_elements"
                        ],
                        expected_elements=evidence["expected_elements"],
                        completion_basis="Shared array acceptance criterion; unique successful contexts and exit-zero PBS logs; every tracked parent inactive",
                    )
                    entry.pop("last_error", None)
                else:
                    raise ValueError(
                        f"Only {evidence['accepted_successful_elements']}/{evidence['expected_elements']} elements verified"
                    )
                self.store.save()
                self.event("reconciled", entry)
            except (ValueError, KeyError, OSError) as error:
                self.failure(entry, "reconciliation", error, block=True)

    def plot(self, entry, directory, *, background=None, members=None):
        if "pending_plot" not in entry:
            entry["pending_plot"] = {
                "created_at": now(),
                "previous_directories": [
                    str(p) for p in directory.glob("*_run_of_create_plots.py_*")
                ]
                if members is not None
                else [],
            }
            self.store.save()
        result = self.backend.plot(
            entry, directory, background=background, members=members
        )
        entry.pop("pending_plot")
        return result

    def single_plots(self):
        for entry in self.store.data["submissions"]:
            if entry["status"] != "finished" or frozen(entry) or protected(entry):
                continue
            try:
                directory = output_path(self.store.root, entry)
                if any(
                    self.scheduler.snapshot.live(
                        a["job_ids"] + a.get("auxiliary_job_ids", [])
                    )
                    for a in entry.get("attempts", [])
                ):
                    continue
                if not entry.get("completion_evidence"):
                    evidence = completion_evidence(
                        directory, entry["attempts"][-1]["job_ids"]
                    )
                    if not evidence["accepted"]:
                        raise ValueError(
                            "Finished entry lacks accepted completion evidence"
                        )
                    entry["completion_evidence"] = evidence
                    self.store.save()
                if not entry.get("single_run_plot"):
                    products = self.plot(entry, directory)
                    entry["single_run_plot"] = {
                        "completed_at": now(),
                        "verified_percentile_plots": products,
                    }
                    self.store.save()
                    self.event("single_plot", entry)
                if entry["completion_evidence"]["complete"] and not entry.get(
                    "intermediate_prune"
                ):
                    self.backend.prune(directory)
                    entry["intermediate_prune"] = {
                        "completed_at": now(),
                        "verified": True,
                        "evidence": entry["completion_evidence"],
                    }
                    self.store.save()
                    self.event("pruned", entry)
                entry["status"] = "analyzed"
                entry.pop("last_error", None)
                self.store.save()
            except (ValueError, OSError, RuntimeError) as error:
                self.failure(entry, "single_run_plot", error)

    def review_tree(self, group, directory):
        saved = {
            (self.store.root / x["remote_submission_directory"]).resolve()
            for x in self.store.data["submissions"]
            if x.get("remote_submission_directory") and not frozen(x)
        }
        for child in directory.glob("*_run_of_submit_train.py_*"):
            if child.resolve() in saved:
                continue
            context = json.loads((child / "context.json").read_text())
            if context.get("run_successful") is False and not context.get(
                "qsub_submissions"
            ):
                record = {
                    "path": str(child),
                    "at": now(),
                    "basis": "false context and empty qsub history before explicit-membership aggregate",
                }
                group.setdefault("stale_residue_cleanup", []).append(record)
                self.store.save()
                shutil.rmtree(child)
                record["verified_removed_at"] = now()
                self.store.save()
            elif context.get("run_successful") is not True:
                raise ValueError(f"Unreviewed failed submission residue: {child}")

    def groups(self):
        for group in self.store.data["plot_groups"]:
            if group["status"] in ("analyzed", "retired") or frozen(group):
                continue
            members = [
                self.by_id[sid]
                for sid in [
                    group["background_submission"],
                    *group["signal_submissions"],
                ]
            ]
            if not all(x["status"] == "analyzed" for x in members) or any(
                self.scheduler.snapshot.live(
                    attempt["job_ids"] + attempt.get("auxiliary_job_ids", [])
                )
                for x in members
                for attempt in x.get("attempts", [])
            ):
                continue
            try:
                tree = (self.store.root / group["remote_multi_run_directory"]).resolve()
                tree.relative_to(self.store.root)
                self.review_tree(group, tree)
                for entry in members:
                    if entry.get("artifact_archive") and not entry.get(
                        "aggregate_restore"
                    ):
                        self.backend.restore(output_path(self.store.root, entry))
                        entry["aggregate_restore"] = {"completed_at": now()}
                        self.store.save()
                products = self.plot(
                    group,
                    tree,
                    background=output_path(self.store.root, members[0]),
                    members=[output_path(self.store.root, x) for x in members[1:]],
                )
                group.update(
                    status="analyzed", completed_at=now(), verified_plots=products
                )
                group.pop("last_error", None)
                self.store.save()
                self.event("aggregate", group)
            except (ValueError, OSError, RuntimeError) as error:
                group["status"] = "failed"
                self.failure(group, "aggregate", error)

    def archives(self):
        for entry in self.store.data["submissions"]:
            if (
                entry["status"] != "analyzed"
                or frozen(entry)
                or protected(entry)
                or entry.get("artifact_archive")
            ):
                continue
            dependencies = [
                g
                for g in self.store.data["plot_groups"]
                if g["status"] != "retired"
                and entry["id"]
                in [g["background_submission"], *g["signal_submissions"]]
            ]
            if any(g["status"] != "analyzed" for g in dependencies):
                continue
            try:
                directory = output_path(self.store.root, entry)
                if entry.get("completion_evidence") and not entry[
                    "completion_evidence"
                ].get("accepted"):
                    raise ValueError(
                        "Archive entry does not meet shared completion criterion"
                    )
                if not directory.exists():
                    self.report["deferred"].append(
                        {
                            "id": entry["id"],
                            "reason": "Saved directory missing; preserve historic outer-archive evidence",
                        }
                    )
                    continue
                if not entry.get("completion_evidence") and not entry.get(
                    "intermediate_prune", {}
                ).get("verified"):
                    evidence = completion_evidence(
                        directory, entry["attempts"][-1]["job_ids"]
                    )
                    if not evidence["accepted"]:
                        raise ValueError("Archive lacks accepted completion evidence")
                    entry["completion_evidence"] = evidence
                    self.store.save()
                if any(
                    self.scheduler.snapshot.live(
                        a["job_ids"] + a.get("auxiliary_job_ids", [])
                    )
                    for a in entry.get("attempts", [])
                ):
                    raise ValueError("Cannot archive active scheduler elements")
                evidence = self.backend.archive(directory)
                entry["artifact_archive"] = {
                    **evidence,
                    "verified_dependent_groups": [g["id"] for g in dependencies],
                    "completion_evidence": entry.get(
                        "completion_evidence", entry.get("intermediate_prune")
                    ),
                }
                entry.pop("aggregate_restore", None)
                entry.pop("last_error", None)
                self.store.save()
                self.event("archived", entry)
            except (ValueError, OSError, RuntimeError) as error:
                self.failure(entry, "artifact_archive", error)

    def submissions(self, manager):
        threshold = None
        entries = self.store.data["submissions"]
        continuations = [
            x
            for x in entries
            if x["status"] == "continuation_requested" and not frozen(x)
        ]
        for entry in continuations + [
            x
            for x in entries
            if x["status"] in ("requested", "blocked") and not frozen(x)
        ]:
            if entry["status"] == "blocked":
                if entry.get("last_error", {}).get("stage") in (
                    "submission",
                    "submission_recovery",
                ):
                    self.report["deferred"].append(
                        {
                            "id": entry["id"],
                            "reason": "Earlier submission failure holds later requests",
                        }
                    )
                    break
                continue
            try:
                config, _ = self.backend.configs(entry)
                if threshold is not None and config.cluster__qsub_n_jobs >= threshold:
                    self.report["quota_skipped"].append(entry["id"])
                    continue
                rejected_size = manager.submit(entry)
                if rejected_size is not None:
                    threshold = rejected_size
            except PreparationRequired as error:
                self.report["requires_review"].append(str(error))
                self.report["deferred"].append(
                    {"id": entry["id"], "reason": str(error)}
                )
                break
            except (ValueError, OSError, RuntimeError) as error:
                self.failure(entry, "submission", error, block=True)
                break

    def run(
        self,
        *,
        archive_completed=False,
        dry_run=False,
        phase="all",
        prepared_commit=None,
    ):
        started = time.monotonic()
        snapshot = self.scheduler.refresh()  # No state save if scheduler access fails.
        if dry_run:
            return {
                "dry_run": True,
                "scheduler_counts": snapshot.counts,
                "status_counts": dict(
                    Counter(x["status"] for x in self.store.data["submissions"])
                ),
                "timings": self.store.timings,
            }
        if prepared_commit is not None:
            if not snapshot.idle or self.backend.git("status", "--porcelain"):
                raise ValueError(
                    "Checkout preparation requires an idle scheduler and clean checkout"
                )
            if (
                self.backend.git("rev-parse", "HEAD") != prepared_commit
                or self.backend.git("rev-parse", "origin/main") != prepared_commit
                or self.backend.git("branch", "--show-current") != "main"
            ):
                raise ValueError(
                    "Prepared checkout must be clean main at verified origin/main"
                )
            owner = json.loads(
                self.store.path.with_suffix(".lock").joinpath("owner.json").read_text()
            )
            for entry in self.store.data["submissions"]:
                if entry["status"] in (
                    "requested",
                    "continuation_requested",
                ) and entry.get("required_pre_submission_action"):
                    entry["pre_submission_action_verification"] = {
                        "at": now(),
                        "remote_commit": prepared_commit,
                        "branch": "main",
                        "lock_acquired_at": owner["acquired_at"],
                        "transport": "Operator-verified repository GitHub App fetch and fast-forward",
                    }
            self.store.data["remote_checkout"] = {
                "branch": "main",
                "commit": prepared_commit,
                "observed_at": now(),
                "latest_main_checked_at": now(),
            }
            self.store.save()
        if archive_completed:
            pending = self.store.data.get("last_v8_configuration_update", {}).get(
                "pending_plot06_calibration", {}
            )
            if pending and pending.get("status") != "complete":
                raise ValueError(
                    "Complete the pending scientific follow-up before resetting state"
                )
            self.report["state_archive"] = self.store.archive_completed(snapshot.idle)
            self.by_id = {x["id"]: x for x in self.store.data["submissions"]}
        manager = SubmissionManager(
            self.store, self.scheduler, self.backend, self.report
        )
        for entry in self.store.data["submissions"]:
            if (
                entry.get("pending_submission_intent")
                and entry["status"] != "retired"
                and not frozen(entry)
            ):
                try:
                    manager.recover(entry)
                except (ValueError, OSError, RuntimeError) as error:
                    self.failure(entry, "submission_recovery", error, block=True)
                    return self.finish(started)
        self.reconcile()
        if phase != "submit":
            for label, action in (
                ("single_plots", self.single_plots),
                ("groups", self.groups),
                ("archives", self.archives),
            ):
                stage_started = time.monotonic()
                action()
                self.report["timings"].append(
                    {"stage": label, "seconds": time.monotonic() - stage_started}
                )
        followup = self.store.data.get("last_v8_configuration_update", {}).get(
            "pending_plot06_calibration", {}
        )
        if followup and followup.get("status") != "complete":
            group = next(
                (
                    g
                    for g in self.store.data["plot_groups"]
                    if g["id"] == followup["wait_for_group"]
                ),
                None,
            )
            if group and group["status"] == "analyzed":
                self.report["requires_review"].append(
                    "Inspect completed Plot06 figure and run the recorded continuous calibration follow-up"
                )
        if phase != "plots":
            self.submissions(manager)
        return self.finish(started)

    def finish(self, started):
        snapshot = (
            self.scheduler.refresh()
            if self.report["events"] or self.report["submissions"]
            else self.scheduler.snapshot
        )
        self.report.update(
            completed_at=now(),
            scheduler_counts=snapshot.counts,
            scheduler_queries=self.scheduler.queries,
            elapsed_seconds=time.monotonic() - started,
            status_counts=dict(
                Counter(x["status"] for x in self.store.data["submissions"])
            ),
        )
        self.store.data["scheduler_snapshot"] = {
            "observed_at": now(),
            "counts": snapshot.counts,
        }
        self.store.data["last_checked_at"] = now()
        self.store.data["last_routine_summary"] = {
            key: self.report[key]
            for key in (
                "completed_at",
                "scheduler_counts",
                "elapsed_seconds",
                "errors",
                "requires_review",
                "submissions",
                "status_counts",
            )
        }
        self.store.save()
        self.report.update(
            state_timings=self.store.timings, state_saves=self.store.saves
        )
        return self.report
