"""Process-level checks for the shared cluster transaction lock."""

import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HELPER = (
    Path(__file__).resolve().parents[1]
    / ".agents/scripts/with_submission_state_lock.py"
)
spec = importlib.util.spec_from_file_location("submission_state_lock", HELPER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SubmissionStateLockTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        (self.root / ".agents").mkdir()
        self.lock = self.root / ".agents/submission-state.lock"

    def test_command_receives_canonical_path_and_releases_after_failure(self):
        output = self.root / "observed.json"
        code = "import json, os, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(json.dumps([os.getcwd(), os.environ['SUBMISSION_STATE_PATH']])); sys.exit(9)"
        self.assertEqual(
            module.run_locked(self.root, [sys.executable, "-c", code, str(output)]), 9
        )
        self.assertEqual(
            json.loads(output.read_text()),
            [str(self.root), str(self.root / ".agents/submission-state.yaml")],
        )
        self.assertFalse(self.lock.exists())

    def test_competing_process_cannot_run_until_holder_exits(self):
        environment = dict(os.environ, WIS_CLUSTER_REMOTE_PROJECT_ROOT=str(self.root))
        holder = subprocess.Popen(
            [
                sys.executable,
                str(HELPER),
                "--",
                sys.executable,
                "-c",
                "print('ready', flush=True); input()",
            ],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "ready")
            owner = json.loads((self.lock / "owner.json").read_text())
            self.assertEqual(owner["pid"], holder.pid)
            marker = self.root / "unexpected"
            self.assertEqual(
                module.run_locked(
                    self.root,
                    [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"],
                ),
                75,
            )
            self.assertFalse(marker.exists())
        finally:
            holder.communicate("\n", timeout=10)
        self.assertFalse(self.lock.exists())
        self.assertEqual(
            module.run_locked(self.root, [sys.executable, "-c", "pass"]), 0
        )

    def test_abandoned_lock_is_not_stolen(self):
        self.lock.mkdir()
        self.assertEqual(
            module.run_locked(self.root, [sys.executable, "-c", "pass"]), 75
        )
        self.assertTrue(self.lock.exists())

    def test_failed_launch_releases_lock(self):
        with self.assertRaises(FileNotFoundError):
            module.run_locked(self.root, [str(self.root / "missing-command")])
        self.assertFalse(self.lock.exists())

    def test_timeout_stops_term_resistant_descendant_before_release(self):
        code = (
            "import os, signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "os.fork(); time.sleep(30)"
        )
        self.assertEqual(
            module.run_locked(
                self.root,
                [sys.executable, "-c", code],
                timeout_seconds=0.3,
                termination_grace_seconds=0.2,
            ),
            124,
        )
        self.assertFalse(self.lock.exists())

    def test_hangup_stops_child_and_releases(self):
        environment = dict(os.environ, WIS_CLUSTER_REMOTE_PROJECT_ROOT=str(self.root))
        holder = subprocess.Popen(
            [
                sys.executable,
                str(HELPER),
                "--",
                sys.executable,
                "-c",
                "import time; print('ready', flush=True); time.sleep(30)",
            ],
            env=environment,
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertEqual(holder.stdout.readline().strip(), "ready")
            holder.send_signal(signal.SIGHUP)
            self.assertEqual(holder.wait(timeout=15), 128 + signal.SIGHUP)
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait()
            holder.stdout.close()
        self.assertFalse(self.lock.exists())

    def test_uncertain_child_liveness_keeps_lock(self):
        with (
            patch.object(module, "group_alive", side_effect=OSError("ps unavailable")),
            self.assertRaises(OSError),
        ):
            module.run_locked(self.root, [sys.executable, "-c", "pass"])
        self.assertTrue((self.lock / "owner.json").exists())

    def test_cli_timeout_and_invalid_timeout(self):
        environment = dict(os.environ, WIS_CLUSTER_REMOTE_PROJECT_ROOT=str(self.root))
        for value, expected in (("0.2", 124), ("0", 2), ("nan", 2), ("inf", 2)):
            with self.subTest(value=value):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(HELPER),
                        "--timeout-seconds",
                        value,
                        "--",
                        sys.executable,
                        "-c",
                        "import time; time.sleep(30)",
                    ],
                    env=environment,
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                self.assertEqual(result.returncode, expected)
                self.assertFalse(self.lock.exists())

    def test_cli_requires_root_and_command(self):
        environment = dict(os.environ)
        environment.pop("WIS_CLUSTER_REMOTE_PROJECT_ROOT", None)
        self.assertEqual(
            subprocess.run(
                [sys.executable, str(HELPER), "--", "true"],
                env=environment,
                capture_output=True,
                check=False,
            ).returncode,
            2,
        )
        environment["WIS_CLUSTER_REMOTE_PROJECT_ROOT"] = str(self.root)
        self.assertEqual(
            subprocess.run(
                [sys.executable, str(HELPER)],
                env=environment,
                capture_output=True,
                check=False,
            ).returncode,
            2,
        )


class ClusterRootExportTest(unittest.TestCase):
    def test_both_ssh_entry_points_export_the_configured_root(self):
        # Exercise command construction without opening another SSH connection.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".gsd").mkdir()
            remote_root = "/cluster/checkout's space"
            (root / ".gsd/SECRETS.md").write_text(
                f"WIS_CLUSTER_SSH_TARGET=agent@cluster\nWIS_CLUSTER_REMOTE_PROJECT_ROOT={remote_root}\n"
            )
            ssh = root / "ssh"
            ssh.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
            ssh.chmod(0o755)
            helper = (
                HELPER.parent.parent / "skills/ssh-to-cluster/scripts/ssh-to-cluster.sh"
            )
            for args in ([], ["pwd"]):
                with self.subTest(interactive=not args):
                    result = subprocess.run(
                        ["sh", str(helper), *args],
                        cwd=root,
                        env=dict(os.environ, PATH=f"{root}:{os.environ['PATH']}"),
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    remote_command = result.stdout.splitlines()[-1]
                    bootstrap = shlex.split(remote_command)[-1]
                    exported = shlex.split(bootstrap.split(";", 1)[0])
                    self.assertEqual(
                        exported,
                        ["export", f"WIS_CLUSTER_REMOTE_PROJECT_ROOT={remote_root}"],
                    )


if __name__ == "__main__":
    unittest.main()
