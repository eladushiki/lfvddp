"""Read-only checkout guards shared by fresh runs, continuations and commit checks."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from frame.git_tools import (
    COMMIT_HASH_ENVIRONMENT_VARIABLE,
    DirtyGitHeadError,
    is_git_head_clean,
    require_clean_git_head,
)


class GitGuardTest(unittest.TestCase):
    def test_real_checkout_ignored_cache_and_changes_without_index_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", directory, *args], stderr=subprocess.DEVNULL
                )

            git("init")
            (root / ".gitignore").write_text("__pycache__/\n")
            (root / "source.py").write_text("original\n")
            git("add", ".")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.org",
                "commit",
                "-m",
                "initial",
            )
            (root / "__pycache__").mkdir()
            (root / "__pycache__/source.pyc").write_bytes(b"cache")
            with patch.dict(os.environ):
                os.environ.pop(COMMIT_HASH_ENVIRONMENT_VARIABLE, None)
                original = Path.cwd()
                os.chdir(root)
                try:
                    index = root / ".git/index"
                    before = (index.read_bytes(), index.stat().st_mtime_ns)
                    self.assertTrue(is_git_head_clean())
                    require_clean_git_head()
                    self.assertEqual(
                        before, (index.read_bytes(), index.stat().st_mtime_ns)
                    )
                    for kind in ("tracked", "staged", "untracked"):
                        with self.subTest(kind=kind):
                            if kind == "untracked":
                                (root / "extra.py").write_text("extra\n")
                            else:
                                (root / "source.py").write_text("changed\n")
                                if kind == "staged":
                                    git("add", "source.py")
                            self.assertFalse(is_git_head_clean())
                            with self.assertRaisesRegex(
                                DirtyGitHeadError, "source.py|extra.py"
                            ):
                                require_clean_git_head()
                            git("reset", "--hard", "HEAD")
                            git("clean", "-fd")
                finally:
                    os.chdir(original)

    def test_transient_change_is_accepted_only_at_same_head(self):
        for heads, succeeds in [(["sha", "sha"], True), (["sha", "other"], False)]:
            with (
                self.subTest(heads=heads),
                patch.dict(os.environ, clear=True),
                patch("frame.git_tools.get_commit_hash", side_effect=heads),
                patch(
                    "frame.git_tools.git_head_changes", side_effect=["?? transient", ""]
                ),
            ):
                if succeeds:
                    require_clean_git_head()
                else:
                    with self.assertRaisesRegex(DirtyGitHeadError, "HEAD changed"):
                        require_clean_git_head()

    def test_worker_commit_requires_no_git_metadata(self):
        with (
            patch.dict(os.environ, {COMMIT_HASH_ENVIRONMENT_VARIABLE: "sha"}),
            patch("frame.git_tools.subprocess.check_output") as command,
        ):
            require_clean_git_head()
            self.assertTrue(is_git_head_clean())
            command.assert_not_called()

    def test_commit_comparison_entrypoint_requires_clean_checkout(self):
        from frame.cluster.is_same_as_commit import main

        for expected, result in [("sha", False), ("different", True)]:
            with (
                self.subTest(expected=expected),
                patch.dict(os.environ, clear=True),
                patch("sys.argv", ["is_same_as_commit", "--commit-hash", expected]),
                patch("frame.git_tools.git_head_changes", return_value=""),
                patch("frame.git_tools.get_commit_hash", return_value="sha"),
                patch(
                    "frame.cluster.is_same_as_commit.get_commit_hash",
                    return_value="sha",
                ),
            ):
                self.assertEqual(main(), result)
        with (
            patch.dict(os.environ, clear=True),
            patch("sys.argv", ["is_same_as_commit", "--commit-hash", "sha"]),
            patch("frame.git_tools.git_head_changes", return_value=" M source.py"),
        ):
            with self.assertRaisesRegex(RuntimeError, "not clean"):
                main()
