#!/usr/bin/env bash
# Source this file from any directory to activate the project Python environment.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
eval "$(cd "$PROJECT_ROOT" && python -c 'from frame.python_environment import cvmfs_python_activation_command; print(cvmfs_python_activation_command())')"
source "$PROJECT_ROOT/.venv/bin/activate"
