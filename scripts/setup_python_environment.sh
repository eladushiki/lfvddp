#!/usr/bin/env bash
# Source this file from the project root to create and activate the locked venv.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(cd "$PROJECT_ROOT" && python -c 'from frame.python_environment import cvmfs_python_activation_command; print(cvmfs_python_activation_command())')"
CVMFS_PYTHON="$(command -v python)"
"$CVMFS_PYTHON" -m pip install --user --upgrade uv
"$CVMFS_PYTHON" -m uv venv --system-site-packages "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"
"$CVMFS_PYTHON" -m uv sync --locked --active
