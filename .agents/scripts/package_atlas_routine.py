#!/usr/bin/env python3
"""Package only committed runner files for an older cluster checkout."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def package(root: Path, output: Path):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    paths = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "frame/cluster/atlas",
            "frame/cluster/submission_intent.py",
            ".agents/scripts/run_atlas_routine.py",
        ],
        cwd=root,
        text=True,
    ).splitlines()
    files = {}
    for path in paths:
        if not path.endswith(".py"):
            continue
        name = (
            "run_atlas_routine.py"
            if path.startswith(".agents/")
            else path.removeprefix("frame/")
        )
        files[name] = subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=root
        )
    if "run_atlas_routine.py" not in files or "cluster/atlas/runner.py" not in files:
        raise ValueError("Commit the runner before packaging it")
    files["manifest.json"] = json.dumps(
        {
            "revision": revision,
            "sha256": {
                name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()
            },
        },
        sort_keys=True,
    ).encode()
    with tarfile.open(output, "w:gz") as archive:
        for name, raw in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(raw))
    print(
        json.dumps(
            {
                "revision": revision,
                "files": len(files),
                "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    package(Path(__file__).resolve().parents[2], args.output)
