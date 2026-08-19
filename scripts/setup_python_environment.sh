#!/usr/bin/env bash
# Source this file from the project root to create and activate the locked venv.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(cd "$PROJECT_ROOT" && python -c 'from frame.python_environment import cvmfs_python_activation_command; print(cvmfs_python_activation_command())')"
UV_CACHE_DIR="${UV_CACHE_DIR:-$(cd "$PROJECT_ROOT" && python -c 'from frame.python_environment import DEFAULT_UV_CACHE_DIR; print(DEFAULT_UV_CACHE_DIR)')}"
export UV_CACHE_DIR
CVMFS_PYTHON="$(command -v python)"
"$CVMFS_PYTHON" -m venv --system-site-packages "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"
python -m pip install --upgrade uv
python -m uv sync --locked --active
