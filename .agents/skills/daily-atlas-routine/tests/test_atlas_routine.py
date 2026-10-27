"""Offline workflow regressions; no SSH, PBS or model training required."""

import gzip
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml
from atlas_routine.artifacts import Artifacts
from atlas_routine.backend import Backend
from atlas_routine.completion import accepted_size, completion_evidence, frozen
from atlas_routine.files import atomic_json
from atlas_routine.runner import Runner
from atlas_routine.scheduler import Scheduler, Snapshot
from atlas_routine.state import StateStore
from atlas_routine.submission import SubmissionManager
from atlas_routine.submission_intent import RECEIPT_ENV
from atlas_routine.submit_entrypoint import install_receipts

from frame.context.run_descriptor import build_run_descriptor

FIXTURE = Path(__file__).parent / "configs/atlas_routine_state.yaml"
SKILL = Path(__file__).resolve().parents[1]
ROOT = SKILL.parents[2]


class AtlasCase(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.path = self.root / ".agents/submission-state.yaml"
        self.path.parent.mkdir()
        self.path.write_bytes(FIXTURE.read_bytes())
        self.lock = self.path.with_suffix(".lock")
        self.lock.mkdir()
        atomic_json(
            self.lock / "owner.json",
            {
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "acquired_at": "test-transaction",
            },
        )
        self.environment = patch.dict(
            os.environ,
            {
                "SUBMISSION_STATE_PATH": str(self.path),
                "WIS_CLUSTER_REMOTE_PROJECT_ROOT": str(self.root),
                "LFVDDP_ATLAS_RUNNER": str(
                    ROOT
                    / ".agents/skills/daily-atlas-routine/scripts/run_atlas_routine.py"
                ),
            },
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.store = StateStore(self.root)

    def entry(self, name="request", status="requested"):
        value = {
            "id": name,
            "status": status,
            "output_root": "results/example",
            "config_packs": [
                "configs/example/generic",
                "configs/example/mandatory-optional/2d/generated",
                "configs/example/plot-05-2d/signal",
            ],
            "only_train": True,
            "debug": False,
            "plot_groups": [],
        }
        self.store.data["submissions"].append(value)
        return value

    def directory(self, entry, expected=10):
        directory = (
            self.root
            / entry["output_root"]
            / build_run_descriptor("stamp", entry["id"], "submit_train.py", 1)
        )
        directory.mkdir(parents=True)
        (directory / "configs").mkdir()
        atomic_json(
            directory / "context.json",
            {
                "run_successful": True,
                "commit_hash": "sha",
                "config": {
                    "cluster__qsub_n_jobs": expected,
                    "cluster__qsub_walltime": "12:00:00",
                },
                "qsub_submissions": [{"job_id": "123[]"}],
            },
        )
        entry["remote_submission_directory"] = str(directory)
        entry["attempts"] = [{"kind": "initial", "job_ids": ["123[]"]}]
        return directory

    def workers(self, directory, count, expected=10):
        for index in range(1, count + 1):
            worker = directory / build_run_descriptor(
                "stamp", "tag", "single_train.py", index
            )
            worker.mkdir()
            atomic_json(
                worker / "context.json",
                {
                    "array_index": index,
                    "run_successful": True,
                    "time": "2026-10-23T01:00:00",
                },
            )
            (directory / f"123[{index}].pbs.OU").write_text("Job exit status: 0\n")

    def scheduler(self, active=()):
        scheduler = Scheduler(command=Mock(return_value=""))
        scheduler.snapshot = Snapshot(list(active), [])
        return scheduler

    def backend(self):
        backend = Mock()
        backend.runtime = self.root / "logs"
        backend.runtime.mkdir(exist_ok=True)
        backend.configs.return_value = (SimpleNamespace(cluster__qsub_n_jobs=100), [])
        backend.prepare.return_value = ("main", "commit")
        return backend


class LockedStateTest(AtlasCase):
    def test_reset_preserves_verified_history_and_global_policy(self):
        self.entry(status="analyzed")
        self.store.save()
        before = self.path.read_bytes()
        evidence = self.store.archive_completed(True)
        self.assertEqual(gzip.decompress(Path(evidence["path"]).read_bytes()), before)
        self.assertEqual(evidence["sha256"], hashlib.sha256(before).hexdigest())
        result = yaml.safe_load(self.path.read_text())
        self.assertEqual(result["submissions"], [])
        self.assertEqual(result["completed_scientific_update"]["status"], "complete")
        self.assertIn("worker_retention_policy", result)

    def test_reset_refuses_pending_work_and_live_scheduler(self):
        entry = self.entry()
        with self.assertRaises(ValueError):
            self.store.archive_completed(True)
        entry["status"] = "analyzed"
        with self.assertRaises(ValueError):
            self.store.archive_completed(False)
        entry["pending_submission_intent"] = {"id": "uncertain"}
        with self.assertRaises(ValueError):
            self.store.archive_completed(True)

    def test_missing_state_never_recreated(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            StateStore(self.root)
        self.assertFalse(self.path.exists())

    def test_unrelated_lock_owner_and_wrong_path_are_rejected(self):
        atomic_json(
            self.lock / "owner.json", {"host": "different-host", "pid": os.getpid()}
        )
        with self.assertRaises(ValueError):
            StateStore(self.root)
        with patch.dict(
            os.environ, {"SUBMISSION_STATE_PATH": str(self.root / "other.yaml")}
        ):
            with self.assertRaises(ValueError):
                StateStore(self.root)

    def test_failed_publish_and_outside_writer_do_not_overwrite_state(self):
        before = self.path.read_bytes()
        self.entry()
        with patch(
            "atlas_routine.state.atomic_bytes", side_effect=OSError("disk full")
        ):
            with self.assertRaises(OSError):
                self.store.save()
        self.assertEqual(self.path.read_bytes(), before)
        self.path.write_bytes(before + b"\n# unexpected writer\n")
        with self.assertRaises(RuntimeError):
            self.store.save()

    def test_shared_threshold_is_strict_and_uses_original_denominator(self):
        self.assertFalse(accepted_size(90, 100))
        self.assertTrue(accepted_size(91, 100))
        entry = self.entry(status="submitted")
        directory = self.directory(entry)
        self.workers(directory, 9)
        evidence = completion_evidence(directory, ["123[]"])
        self.assertEqual(evidence["expected_elements"], 10)
        self.assertFalse(evidence["accepted"])
        self.workers_missing_one(directory)
        evidence = completion_evidence(directory, ["123[]"])
        self.assertTrue(evidence["complete"])

    def workers_missing_one(self, directory):
        worker = directory / build_run_descriptor("stamp", "tag", "single_train.py", 10)
        worker.mkdir()
        atomic_json(
            worker / "context.json", {"array_index": 10, "run_successful": True}
        )
        (directory / "123[10].pbs.OU").write_text("Job exit status: 0\n")

    def test_duplicate_successful_workers_are_not_double_counted(self):
        entry = self.entry()
        directory = self.directory(entry)
        self.workers(directory, 1)
        worker = directory / "other_run_of_single_train.py_pid_99"
        worker.mkdir()
        atomic_json(worker / "context.json", {"array_index": 1, "run_successful": True})
        with self.assertRaises(ValueError):
            completion_evidence(directory, ["123[]"])

    def test_malformed_context_and_nonzero_log_preserved_in_evidence(self):
        entry = self.entry()
        directory = self.directory(entry, expected=100)
        self.workers(directory, 99)
        broken = directory / "stamp_run_of_single_train.py_pid_100"
        broken.mkdir()
        (broken / "context.json").write_text("{")
        (directory / "123[100].pbs.OU").write_text("Job exit status: 265\n")
        evidence = completion_evidence(directory, ["123[]"])
        self.assertTrue(evidence["accepted"])
        self.assertFalse(evidence["complete"])
        self.assertEqual(evidence["excluded_indices"], [100])
        self.assertEqual(len(evidence["invalid_contexts"]), 1)
        self.assertIn("265", evidence["failed_logs"][0]["text"])

    def test_active_array_never_reads_outputs(self):
        entry = self.entry(status="submitted")
        entry["attempts"] = [{"job_ids": ["123[]"]}]
        scheduler = self.scheduler([["123[1]", "user", "R", "q"]])
        runner = Runner(self.store, scheduler, self.backend())
        runner.reconcile()
        self.assertEqual(entry["status"], "submitted")
        self.assertNotIn("completion_evidence", entry)

    def test_walltime_kill_requests_continuation_before_acceptance(self):
        entry = self.entry(status="submitted")
        directory = self.directory(entry)
        self.workers(directory, 9)
        (directory / "123[10].pbs.OU").write_text(
            "PBS: job killed: walltime\nJob exit status: 271\n"
        )
        Runner(self.store, self.scheduler(), self.backend()).reconcile()
        self.assertEqual(entry["status"], "continuation_requested")
        self.assertEqual(entry["pending_continuation"]["extra_time"], "12:00:00")

    def test_only_complete_declared_group_runs_and_is_idempotent(self):
        background = self.entry("background", "analyzed")
        signal = self.entry("signal", "requested")
        bg = self.directory(background)
        sg = self.directory(signal)
        group = {
            "id": "group",
            "status": "pending",
            "background_submission": "background",
            "signal_submissions": ["signal"],
            "remote_multi_run_directory": str(sg.parent),
        }
        self.store.data["plot_groups"] = [group]
        backend = self.backend()
        backend.plot.return_value = ["verified.png"]
        runner = Runner(self.store, self.scheduler(), backend)
        runner.groups()
        backend.plot.assert_not_called()
        signal["status"] = "analyzed"
        runner.groups()
        self.assertEqual(group["status"], "analyzed")
        self.assertEqual(backend.plot.call_args.kwargs["background"], bg)
        self.assertEqual(backend.plot.call_args.kwargs["members"], [sg])
        runner.groups()
        self.assertEqual(backend.plot.call_count, 1)

    def test_partial_array_is_plotted_but_not_pruned(self):
        entry = self.entry(status="finished")
        self.directory(entry)
        entry["completion_evidence"] = {"complete": False, "accepted": True}
        backend = self.backend()
        backend.plot.return_value = ["progression.png"]
        Runner(self.store, self.scheduler(), backend).single_plots()
        self.assertEqual(entry["status"], "analyzed")
        backend.artifacts.prune.assert_not_called()

    def test_frozen_and_protected_outputs_not_touched(self):
        entry = self.entry(status="finished")
        entry["frozen"] = True
        runner = Runner(self.store, self.scheduler(), self.backend())
        self.assertTrue(frozen(entry))
        runner.single_plots()
        runner.backend.plot.assert_not_called()
        entry["config_packs"] = ["configs/example/generic"]
        entry["frozen"] = False
        entry["purpose"] = "research workflow"
        runner.single_plots()
        runner.backend.plot.assert_not_called()

    def test_continuations_first_and_quota_threshold_only_skips_large_arrays(self):
        [
            self.entry(name, status)
            for name, status in [
                ("large", "requested"),
                ("equal", "requested"),
                ("small", "requested"),
                ("continuation", "continuation_requested"),
            ]
        ]
        backend = self.backend()
        backend.configs.side_effect = lambda entry: (
            SimpleNamespace(cluster__qsub_n_jobs=10 if entry["id"] == "small" else 100),
            [],
        )
        manager = Mock()
        manager.submit.side_effect = lambda entry: (
            100 if entry["id"] == "large" else None
        )
        runner = Runner(self.store, self.scheduler(), backend)
        runner.submissions(manager)
        self.assertEqual(
            [call.args[0]["id"] for call in manager.submit.call_args_list],
            ["continuation", "large", "small"],
        )
        self.assertEqual(runner.report["quota_skipped"], ["equal"])

    def test_blocked_submission_holds_later_requests(self):
        entry = self.entry("blocked", "blocked")
        entry["last_error"] = {"stage": "submission"}
        self.entry("later")
        manager = Mock()
        Runner(self.store, self.scheduler(), self.backend()).submissions(manager)
        manager.submit.assert_not_called()

    def test_noop_uses_one_snapshot_and_no_output_scans(self):
        scheduler = Scheduler(command=Mock(return_value=""))
        backend = self.backend()
        result = Runner(self.store, scheduler, backend).run()
        self.assertEqual(scheduler.queries, 2)
        backend.plot.assert_not_called()
        backend.archive.assert_not_called()
        self.assertEqual(result["events"], [])

    def test_scheduler_failure_leaves_timestamp_and_bytes_unchanged(self):
        before = self.path.read_bytes()
        scheduler = Scheduler(
            command=Mock(side_effect=subprocess.TimeoutExpired("qstat", 40))
        )
        with self.assertRaises(subprocess.TimeoutExpired):
            Runner(self.store, scheduler, self.backend()).run()
        self.assertEqual(self.path.read_bytes(), before)

    def test_intent_checkpoint_precedes_launch_and_recovers_without_second_qsub(self):
        entry = self.entry()
        self.store.save()
        backend = self.backend()
        scheduler = self.scheduler()
        scheduler.refresh = Mock(return_value=Snapshot([["123[1]", "u", "R", "q"]], []))
        manager = SubmissionManager(
            self.store, scheduler, backend, {"submissions": [], "quota_rejected": []}
        )

        def launch(args, label, env):
            saved = yaml.safe_load(self.path.read_text())["submissions"][0]
            self.assertIn("pending_submission_intent", saved)
            receipt_path = Path(env[RECEIPT_ENV])
            receipt = json.loads(receipt_path.read_text())
            directory = self.directory(dict(entry), expected=100)
            atomic_json(
                directory / "context.json",
                {
                    "run_successful": False,
                    "commit_hash": "commit",
                    "config": {"cluster__qsub_n_jobs": 100},
                    "qsub_submissions": [],
                },
            )
            receipt["calls"] = [
                {
                    "kind": "train",
                    "status": "accepted",
                    "job_id": "123[]",
                    "context_directory": str(directory),
                    "walltime": "12:00:00",
                }
            ]
            atomic_json(receipt_path, receipt)
            raise InterruptedError("crash after qsub, before state save")

        backend.command.side_effect = launch
        with self.assertRaises(InterruptedError):
            manager.submit(entry)
        backend.command.reset_mock()
        manager.recover(entry)
        backend.command.assert_not_called()
        self.assertEqual(entry["status"], "submitted")
        self.assertEqual(entry["attempts"][-1]["job_ids"], ["123[]"])
        self.assertEqual(
            len(
                json.loads(
                    (
                        Path(entry["remote_submission_directory"]) / "context.json"
                    ).read_text()
                )["qsub_submissions"]
            ),
            1,
        )

    def test_uncertain_receipt_without_scheduler_match_never_resubmits(self):
        entry = self.entry()
        receipt_path = (
            self.root / ".agents/submission-state-tools/intents" / ("a" * 32 + ".json")
        )
        atomic_json(
            receipt_path,
            {
                "id": "a" * 32,
                "calls": [{"kind": "train", "status": "started", "pbs_name": "a123"}],
            },
        )
        entry["pending_submission_intent"] = {
            "id": "a" * 32,
            "receipt_path": str(receipt_path),
        }
        scheduler = self.scheduler()
        scheduler.recover_call = Mock(side_effect=RuntimeError("uncertain"))
        backend = self.backend()
        manager = SubmissionManager(self.store, scheduler, backend, {"submissions": []})
        with self.assertRaises(RuntimeError):
            manager.recover(entry)
        backend.command.assert_not_called()

    def test_cli_dry_run_does_not_mutate_state(self):
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        script = fake_bin / "qstat"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        before = self.path.read_bytes()
        environment = dict(
            os.environ,
            PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
            PYTHONPATH=str(ROOT),
        )
        result = subprocess.run(
            [
                sys.executable,
                str(
                    ROOT
                    / ".agents/skills/daily-atlas-routine/scripts/run_atlas_routine.py"
                ),
                "--held-lock",
                "--dry-run",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(json.loads(result.stdout)["dry_run"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_qsub_receipt_records_before_execution_and_limits_retries(self):
        from atlas_routine.submit_entrypoint import install_receipts

        import frame.submit as module

        path = (
            self.root / ".agents/submission-state-tools/intents" / ("b" * 32 + ".json")
        )
        atomic_json(path, {"id": "b" * 32, "calls": []})
        context = SimpleNamespace(
            unique_out_dir=self.root,
            config=SimpleNamespace(
                cluster__qsub_io=1, cluster__qsub_walltime="12:00:00"
            ),
        )

        def qsub(command, **kwargs):
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["calls"][0]["status"], "started")
            self.assertIn("-N abbbbbbbbbb00", command)
            self.assertIn("LFVDDP_SUBMISSION_INTENT=" + "b" * 32, command)
            return b"456[].server\n"

        original = module.qsub_a_script
        with (
            patch.dict(os.environ, {RECEIPT_ENV: str(path)}),
            patch.object(module, "ClusterConfig", object),
            patch.object(module, "check_output", side_effect=qsub) as execute,
        ):
            try:
                install_receipts(module)
                result = module.qsub_a_script(
                    context, Path("script.sh"), "tag_exec_train"
                )
            finally:
                module.qsub_a_script = original
        self.assertEqual(result, "456[].server")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(json.loads(path.read_text())["calls"][0]["job_id"], result)


class ComparisonGateTest(AtlasCase):
    def test_submission_mismatch_stops_before_intent_or_qsub(self):
        background = self.entry("background")
        signal = self.entry("signal")
        self.store.data["plot_groups"] = [
            {
                "id": "comparison",
                "status": "pending",
                "background_submission": background["id"],
                "signal_submissions": [signal["id"]],
            }
        ]
        backend = self.backend()
        backend.validate_comparison.side_effect = ValueError("train__epochs differs")
        manager = Mock()
        runner = Runner(self.store, self.scheduler(), backend)
        runner.submissions(manager)
        manager.submit.assert_not_called()
        self.assertEqual(background["status"], "blocked")
        self.assertNotIn("pending_submission_intent", background)
        self.assertIn("train__epochs", background["last_error"]["message"])

    def test_aggregate_mismatch_precedes_restore_cleanup_and_plot(self):
        background = self.entry("background", "analyzed")
        signal = self.entry("signal", "analyzed")
        self.directory(background)
        self.directory(signal)
        signal["artifact_archive"] = {"archive_path": "saved.tar.gz"}
        group = {
            "id": "comparison",
            "status": "pending",
            "background_submission": background["id"],
            "signal_submissions": [signal["id"]],
            "remote_multi_run_directory": background["output_root"],
        }
        self.store.data["plot_groups"] = [group]
        backend = self.backend()
        backend.validate_comparison.side_effect = ValueError("train__nuisance differs")
        runner = Runner(self.store, self.scheduler(), backend)
        runner.review_tree = Mock()
        runner.groups()
        runner.review_tree.assert_not_called()
        backend.artifacts.restore.assert_not_called()
        backend.plot.assert_not_called()
        self.assertEqual(group["status"], "failed")
        self.assertNotIn("pending_plot", group)
        self.assertIn("train__nuisance", group["last_error"]["message"])


class SchedulerTest(unittest.TestCase):
    def test_recovery_requires_token_owner_and_unique_parent(self):
        token = "a" * 32
        call = {"pbs_name": "intent-name"}
        jobs = {
            "1[].server": {
                "Job_Name": "intent-name",
                "Job_Owner": os.environ["USER"] + "@host",
                "Variable_List": {"LFVDDP_SUBMISSION_INTENT": token},
            },
            "2[].server": {
                "Job_Name": "unrelated",
                "Job_Owner": os.environ["USER"] + "@host",
                "Variable_List": {"LFVDDP_SUBMISSION_INTENT": token},
            },
        }
        scheduler = Scheduler(command=Mock(return_value=json.dumps({"Jobs": jobs})))
        self.assertEqual(scheduler.recover_call(call, token), "1[]")
        with self.assertRaises(RuntimeError):
            scheduler.recover_call(call, "b" * 32)


class RecoveryAndArchiveTest(AtlasCase):
    def test_recovered_job_must_be_active_or_have_verified_finished_outputs(self):
        entry = self.entry()
        directory = self.directory(entry)
        entry["attempts"] = []
        receipt = {
            "id": "c" * 32,
            "calls": [
                {
                    "kind": "train",
                    "status": "accepted",
                    "job_id": "123[]",
                    "context_directory": str(directory),
                    "walltime": "12:00:00",
                }
            ],
        }
        intent = {
            "id": receipt["id"],
            "kind": "initial",
            "branch": "main",
            "commit": "sha",
            "command": [],
        }
        scheduler = self.scheduler()
        scheduler.refresh = Mock(return_value=Snapshot([], [["123[1]", "u", "F", "q"]]))
        manager = SubmissionManager(
            self.store, scheduler, self.backend(), {"submissions": []}
        )
        with self.assertRaises(RuntimeError):
            manager.finalize(entry, intent, receipt)
        self.assertEqual(entry["attempts"], [])

    def test_prepared_receipt_proves_qsub_not_invoked_and_cleans_only_residue(self):
        entry = self.entry()
        receipt_path = (
            self.root / ".agents/submission-state-tools/intents" / ("d" * 32 + ".json")
        )
        receipt = {"id": "d" * 32, "calls": []}
        atomic_json(receipt_path, receipt)
        directory = self.directory(entry)
        entry["attempts"] = []
        atomic_json(
            directory / "context.json",
            {"run_successful": False, "qsub_submissions": []},
        )
        entry["pending_submission_intent"] = {
            "id": receipt["id"],
            "receipt_path": str(receipt_path),
            "kind": "initial",
            "previous_directories": [],
        }
        manager = SubmissionManager(
            self.store, self.scheduler(), self.backend(), {"submissions": []}
        )
        manager.recover(entry)
        self.assertFalse(directory.exists())
        self.assertNotIn("pending_submission_intent", entry)
        self.assertEqual(entry["status"], "requested")

    def test_quota_failure_retains_request_and_removes_verified_pre_qsub_directory(
        self,
    ):
        entry = self.entry()
        backend = self.backend()

        def reject(args, label, env):
            directory = self.directory(dict(entry), expected=100)
            atomic_json(
                directory / "context.json",
                {"run_successful": False, "qsub_submissions": []},
            )
            receipt_path = Path(env[RECEIPT_ENV])
            receipt = json.loads(receipt_path.read_text())
            receipt["calls"] = [
                {
                    "kind": "train",
                    "status": "rejected",
                    "error": "quota",
                    "context_directory": str(directory),
                }
            ]
            atomic_json(receipt_path, receipt)
            (backend.runtime / (label + ".log")).write_text(
                "qsub: request would exceed per-user limit"
            )
            return SimpleNamespace(returncode=1)

        backend.command.side_effect = reject
        scheduler = self.scheduler()
        scheduler.refresh = Mock(return_value=scheduler.snapshot)
        report = {"submissions": [], "quota_rejected": []}
        result = SubmissionManager(self.store, scheduler, backend, report).submit(entry)
        self.assertEqual(result, 100)
        self.assertEqual(entry["status"], "requested")
        self.assertNotIn("pending_submission_intent", entry)
        self.assertEqual(list((self.root / entry["output_root"]).iterdir()), [])
        self.assertEqual(report["quota_rejected"], [entry["id"]])

    def test_archive_retains_first_worker_regardless_of_debug_and_recovers_cleanup(
        self,
    ):
        for debug in (False, True):
            with self.subTest(debug=debug):
                entry = self.entry("archive-" + str(debug), "analyzed")
                directory = self.directory(entry, expected=2)
                context = json.loads((directory / "context.json").read_text())
                context["is_debug_mode"] = debug
                atomic_json(directory / "context.json", context)
                self.workers(directory, 2)
                for worker in directory.glob("*_run_of_single_train.py_*"):
                    (worker / "prediction.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                report = {"timings": []}
                backend = Backend(self.root, self.root / "runtime", report)
                # Use the actual committed helper, with this test's isolated results.
                helper = Artifacts(ROOT).archive_helper()
                backend.artifacts.helper = helper
                with patch.object(
                    helper.shutil,
                    "rmtree",
                    side_effect=OSError("interruption after verified tar"),
                ):
                    with self.assertRaises(OSError):
                        backend.artifacts.archive(directory)
                self.assertTrue((directory / "array-job-artifacts.tar.gz").exists())
                result = backend.artifacts.archive(directory)
                self.assertEqual(len(result["retained_worker_directories"]), 1)
                self.assertEqual(
                    len(list(directory.glob("*_run_of_single_train.py_*"))), 1
                )
                self.assertTrue(
                    (
                        Path(result["retained_worker_directories"][0])
                        / "prediction.png"
                    ).is_file()
                )

    def test_failed_archive_recovery_preserves_conflicting_sources(self):
        import tarfile

        entry = self.entry("conflict", "analyzed")
        directory = self.directory(entry, expected=2)
        self.workers(directory, 2)
        workers = sorted(directory.glob("*_run_of_single_train.py_*"))
        data = workers[1] / "data.json"
        data.write_text("original")
        with tarfile.open(directory / "array-job-artifacts.tar.gz", "w:gz") as archive:
            archive.add(workers[1], arcname=workers[1].name)
            for log in directory.glob("*.OU*"):
                archive.add(log, arcname=log.name)
        data.write_text("different")
        backend = Backend(self.root, self.root / "runtime", {"timings": []})
        backend.artifacts.helper = Artifacts(ROOT).archive_helper()
        with self.assertRaises(ValueError):
            backend.artifacts.archive(directory)
        self.assertEqual(data.read_text(), "different")

    def test_qsub_rejection_is_recorded_and_never_retried(self):
        import frame.submit as module

        path = (
            self.root / ".agents/submission-state-tools/intents" / ("e" * 32 + ".json")
        )
        atomic_json(path, {"id": "e" * 32, "calls": []})
        context = SimpleNamespace(
            unique_out_dir=self.root,
            config=SimpleNamespace(
                cluster__qsub_io=1, cluster__qsub_walltime="12:00:00"
            ),
        )
        original = module.qsub_a_script
        with (
            patch.dict(os.environ, {RECEIPT_ENV: str(path)}),
            patch.object(module, "ClusterConfig", object),
            patch.object(
                module,
                "check_output",
                side_effect=subprocess.CalledProcessError(
                    1, "qsub", b"qsub: per-user limit"
                ),
            ) as execute,
        ):
            try:
                install_receipts(module)
                with self.assertRaises(RuntimeError):
                    module.qsub_a_script(
                        context, Path("script.sh"), "tag_exec_train", max_tries=3
                    )
            finally:
                module.qsub_a_script = original
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(json.loads(path.read_text())["calls"][0]["status"], "rejected")

    def test_ambiguous_qsub_error_is_recorded_and_never_retried(self):
        import frame.submit as module

        path = (
            self.root / ".agents/submission-state-tools/intents" / ("f" * 32 + ".json")
        )
        atomic_json(path, {"id": "f" * 32, "calls": []})
        context = SimpleNamespace(
            unique_out_dir=self.root,
            config=SimpleNamespace(
                cluster__qsub_io=1, cluster__qsub_walltime="12:00:00"
            ),
        )
        original = module.qsub_a_script
        with (
            patch.dict(os.environ, {RECEIPT_ENV: str(path)}),
            patch.object(module, "ClusterConfig", object),
            patch.object(
                module,
                "check_output",
                side_effect=TimeoutError("scheduler outcome unknown"),
            ) as execute,
        ):
            try:
                install_receipts(module)
                with self.assertRaises(RuntimeError):
                    module.qsub_a_script(context, Path("script.sh"), "tag_exec_train")
            finally:
                module.qsub_a_script = original
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(
            json.loads(path.read_text())["calls"][0]["status"], "uncertain"
        )


class AdditionalWorkflowTest(AtlasCase):
    def test_saved_config_layers_are_passed_in_order_without_name_inference(self):
        entry = self.entry()
        entry["config_packs"] = [
            "configs/base",
            "configs/custom-geometry",
            "configs/override",
            "configs/final",
        ]
        files = [self.root / pack for pack in entry["config_packs"]]
        backend = Backend(self.root, self.root / "runtime", {"timings": []})
        with (
            patch(
                "frame.file_system.textual_data.expand_config_paths", return_value=files
            ) as expand,
            patch(
                "frame.file_system.textual_data.load_config_params_from_paths",
                return_value={},
            ) as load,
            patch(
                "frame.context.execution_context.create_config_from_paramters",
                return_value=object(),
            ) as create,
        ):
            config, actual = backend.configs(entry)
        expand.assert_called_once_with(files)
        load.assert_called_once_with(files)
        self.assertIs(config, create.return_value)
        self.assertEqual(actual, files)

    def test_programming_error_escapes_stage_boundary_without_marking_failed(self):
        entry = self.entry(status="finished")
        self.directory(entry)
        entry["completion_evidence"] = {"complete": True, "accepted": True}
        backend = self.backend()
        backend.plot.side_effect = TypeError("unexpected programming error")
        runner = Runner(self.store, self.scheduler(), backend)
        with self.assertRaises(TypeError):
            runner.single_plots()
        self.assertNotIn("last_error", entry)

    def test_continuation_corrects_only_last_defining_layer(self):
        import shutil

        files = []
        for name in ("atlas_walltime_base.json", "atlas_walltime_override.json"):
            target = self.root / name
            shutil.copyfile(FIXTURE.parent / name, target)
            files.append(target)
        backend = Backend(self.root, self.root / "runtime", {"timings": []})
        backend.configs = Mock(return_value=(None, files))
        original = files[0].read_bytes()
        evidence = backend.correct_walltime({}, "06:00:00")
        self.assertEqual(files[0].read_bytes(), original)
        self.assertEqual(
            json.loads(files[1].read_text())["cluster__qsub_walltime"], "06:00:00"
        )
        self.assertEqual(evidence["before"], "03:00:00")

    def test_one_element_submission_without_array_index(self):
        entry = self.entry()
        directory = self.directory(entry, expected=1)
        worker = directory / build_run_descriptor(
            "stamp", "single", "single_train.py", 1
        )
        worker.mkdir()
        atomic_json(
            worker / "context.json", {"array_index": None, "run_successful": True}
        )
        (directory / "123.pbs.OU").write_text("Job exit status: 0\n")
        self.assertTrue(completion_evidence(directory, ["123"])["complete"])

    def test_preparation_gate_keeps_request_pending_and_does_not_overtake(self):
        from atlas_routine.backend import PreparationRequired

        first = self.entry("first")
        self.entry("later")
        manager = Mock()
        manager.submit.side_effect = PreparationRequired(
            "latest-main preparation required"
        )
        runner = Runner(self.store, self.scheduler(), self.backend())
        runner.submissions(manager)
        self.assertEqual(first["status"], "requested")
        self.assertEqual(manager.submit.call_count, 1)
        self.assertTrue(runner.report["requires_review"])

    def test_plot_phase_never_submits_saved_request(self):
        self.entry()
        scheduler = Scheduler(command=Mock(return_value=""))
        with patch("atlas_routine.runner.SubmissionManager") as manager:
            Runner(self.store, scheduler, self.backend()).run(phase="plots")
        manager.return_value.submit.assert_not_called()

    def test_restore_accepts_identical_retained_copy_and_is_idempotent(self):
        import tarfile

        entry = self.entry("restore", "analyzed")
        directory = self.directory(entry, expected=1)
        retained = directory / build_run_descriptor(
            "stamp", "example", "single_train.py", 1
        )
        retained.mkdir()
        picture = retained / "prediction.png"
        picture.write_bytes(b"\x89PNG\r\n\x1a\n")
        other = directory / "artifact.txt"
        other.write_text("archive content")
        with tarfile.open(directory / "array-job-artifacts.tar.gz", "w:gz") as archive:
            archive.add(retained, arcname=retained.name)
            archive.add(other, arcname=other.name)
        other.unlink()
        backend = Backend(self.root, self.root / "runtime", {"timings": []})
        backend.artifacts.restore(directory)
        backend.artifacts.restore(directory)
        self.assertEqual(other.read_text(), "archive content")
        picture.write_bytes(b"conflict")
        with self.assertRaises(ValueError):
            backend.artifacts.restore(directory)

    def test_server_suffix_does_not_duplicate_context_submission_record(self):
        entry = self.entry()
        directory = self.directory(entry)
        entry["attempts"] = []
        receipt = {
            "calls": [
                {
                    "kind": "train",
                    "status": "accepted",
                    "job_id": "123[].server",
                    "context_directory": str(directory),
                    "walltime": "12:00:00",
                }
            ]
        }
        intent = {
            "id": "a" * 32,
            "kind": "initial",
            "branch": "main",
            "commit": "sha",
            "command": [],
        }
        entry["pending_submission_intent"] = intent
        scheduler = self.scheduler()
        scheduler.refresh = Mock(return_value=Snapshot([["123[1]", "u", "R", "q"]], []))
        manager = SubmissionManager(
            self.store, scheduler, self.backend(), {"submissions": []}
        )
        manager.finalize(entry, intent, receipt)
        self.assertEqual(entry["attempts"][0]["job_ids"], ["123[]"])
        self.assertEqual(
            len(
                json.loads((directory / "context.json").read_text())["qsub_submissions"]
            ),
            1,
        )


class LauncherTest(AtlasCase):
    def test_ssh_helper_exports_configured_root_once_without_a_real_connection(self):
        import shlex

        cluster = self.root / "configured cluster"
        (cluster / ".venv/bin").mkdir(parents=True)
        (cluster / ".venv/bin/activate").write_text(":\n")
        (cluster / "scripts").mkdir()
        (cluster / "scripts/activate_python_environment.sh").write_text(":\n")
        (self.root / ".gsd").mkdir()
        (self.root / ".gsd/SECRETS.md").write_text(
            f"WIS_CLUSTER_SSH_TARGET=test@example.invalid\nWIS_CLUSTER_REMOTE_PROJECT_ROOT={cluster}\n"
        )
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        trace = self.root / "ssh-calls"
        ssh = fake_bin / "ssh"
        ssh.write_text(
            '#!/bin/sh\nprintf x >> "$TEST_SSH_TRACE"\nfor last; do :; done\nexec /bin/bash --noprofile --norc -c "$last"\n'
        )
        ssh.chmod(0o755)
        command = shlex.join(
            [
                sys.executable,
                "-c",
                "import os; print(os.environ['WIS_CLUSTER_REMOTE_PROJECT_ROOT'])",
            ]
        )
        environment = dict(
            os.environ,
            PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
            TEST_SSH_TRACE=str(trace),
        )
        result = subprocess.run(
            [
                str(ROOT / ".agents/skills/ssh-to-cluster/scripts/ssh-to-cluster.sh"),
                command,
            ],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(cluster))
        self.assertEqual(trace.read_text(), "x")

    def test_stale_preparation_does_not_satisfy_new_idle_transaction(self):
        from atlas_routine.backend import PreparationRequired

        entry = self.entry()
        entry["required_pre_submission_action"] = "Fetch and fast-forward latest main"
        entry["pre_submission_action_verification"] = {
            "remote_commit": "sha",
            "lock_acquired_at": "previous-transaction",
        }
        backend = Backend(self.root, self.root / "runtime", {"timings": []})
        backend.git = Mock(side_effect=["main", "", "sha"])
        with self.assertRaises(PreparationRequired):
            backend.prepare(entry, self.store.data, Snapshot([], []))


class DeploymentTest(AtlasCase):
    def test_skill_deployment_verifies_sources_and_rejects_drift(self):
        import shutil

        deployment = self.root / ".agents/submission-state-tools/runner"
        deployment.mkdir(parents=True)
        shutil.copyfile(
            ROOT / ".agents/skills/daily-atlas-routine/scripts/run_atlas_routine.py",
            deployment / "run_atlas_routine.py",
        )
        shutil.copytree(
            SKILL / "scripts/atlas_routine",
            deployment / "atlas_routine",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        hashes = {
            str(p.relative_to(deployment)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in deployment.rglob("*.py")
        }
        atomic_json(
            deployment / "manifest.json",
            {"revision": "fixture-revision", "sha256": hashes},
        )
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        (fake_bin / "qstat").write_text("#!/bin/sh\nexit 0\n")
        (fake_bin / "qstat").chmod(0o755)
        env = dict(
            os.environ,
            PATH=str(fake_bin) + os.pathsep + os.environ["PATH"],
            PYTHONPATH=str(ROOT),
        )
        command = [
            sys.executable,
            str(deployment / "run_atlas_routine.py"),
            "--held-lock",
        ]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["implementation_revision"], "fixture-revision"
        )
        (deployment / "atlas_routine/runner.py").write_text(
            "raise RuntimeError('changed')"
        )
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("checksum", result.stderr)


if __name__ == "__main__":
    unittest.main()


class CleanSubmissionRecoveryTest(AtlasCase):
    def test_clean_bracket_retries_once_before_qsub_with_saved_audit(self):
        for error_type in ("RuntimeError", "DirtyGitHeadError"):
            for kind in ("initial", "continuation"):
                with self.subTest(error=error_type, kind=kind):
                    backend = self.backend()
                    entry = self.entry(name=error_type + kind)
                    receipt_path = self.path.parent / (
                        "receipt-" + entry["id"] + ".json"
                    )
                    atomic_json(receipt_path, {"calls": []})
                    intent = {
                        "receipt_path": str(receipt_path),
                        "branch": "main",
                        "commit": "commit",
                        "kind": kind,
                        "previous_directories": [],
                    }
                    entry["pending_submission_intent"] = intent
                    manager = SubmissionManager(
                        self.store, self.scheduler(), backend, {}
                    )

                    def command(
                        args, label, env, backend=backend, error_type=error_type
                    ):
                        if not label.endswith("-clean-retry"):
                            (backend.runtime / (label + ".log")).write_text(
                                json.dumps(
                                    {
                                        "error": error_type,
                                        "message": "Commit changes before running the script.\n?? transient",
                                    }
                                )
                            )
                            return SimpleNamespace(returncode=2)
                        saved = yaml.safe_load(self.path.read_text())["submissions"][-1]
                        self.assertIn(
                            "cleanliness_recovery", saved["pending_submission_intent"]
                        )
                        self.assertEqual(args[-1], "argument")
                        return SimpleNamespace(returncode=0)

                    backend.command.side_effect = command
                    result, label = manager.launch(
                        entry, intent, {}, ["argument"], "attempt"
                    )
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(label, "attempt-clean-retry")
                    self.assertEqual(backend.command.call_count, 2)
                    backend.prepare.assert_called_once()

    def test_never_retries_qsub_started_other_failure_changed_head_or_dirty(self):
        for mode in (
            "started",
            "uncertain",
            "accepted",
            "rejected",
            "prepared",
            "unrelated",
            "head",
            "dirty",
            "persistent",
        ):
            with self.subTest(mode=mode):
                backend = self.backend()
                entry = self.entry(name=mode)
                path = self.path.parent / (mode + ".json")
                atomic_json(
                    path,
                    {
                        "calls": [{"status": mode}]
                        if mode
                        in ("started", "uncertain", "accepted", "rejected", "prepared")
                        else []
                    },
                )
                intent = {
                    "receipt_path": str(path),
                    "branch": "main",
                    "commit": "commit",
                    "previous_directories": [],
                }
                (backend.runtime / "failure.log").write_text(
                    json.dumps(
                        {
                            "error": "RuntimeError",
                            "message": "other failure"
                            if mode == "unrelated"
                            else "Commit changes before running the script.",
                        }
                    )
                )
                (backend.runtime / "failure-clean-retry.log").write_text(
                    "persistent failure"
                )
                backend.command.return_value = SimpleNamespace(returncode=2)
                if mode == "head":
                    backend.prepare.return_value = ("main", "changed")
                if mode == "dirty":
                    backend.prepare.side_effect = RuntimeError("Checkout is dirty")
                manager = SubmissionManager(self.store, self.scheduler(), backend, {})
                if mode in ("head", "dirty"):
                    with self.assertRaises(RuntimeError):
                        manager.launch(entry, intent, {}, [], "failure")
                else:
                    result, _ = manager.launch(entry, intent, {}, [], "failure")
                    self.assertEqual(result.returncode, 2)
                self.assertEqual(
                    backend.command.call_count, 2 if mode == "persistent" else 1
                )
