#!/usr/bin/env bash
# Source this file from any directory to activate the project Python environment.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
activation_command="$(cd "$PROJECT_ROOT" && python -c 'from frame.python_environment import cvmfs_python_activation_command; print(cvmfs_python_activation_command())')" || return 1
eval "$activation_command" || return 1
source "$PROJECT_ROOT/.venv/bin/activate" || return 1
