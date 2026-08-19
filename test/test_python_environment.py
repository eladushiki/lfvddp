from pathlib import Path

import pytest

from frame.command_line.execution import (
    format_qsub_build_script,
    format_qsub_execution_script,
)
from frame.python_environment import (
    CVMFS_PYTHON_SETUP_PATH,
    cvmfs_python_activation_command,
    python_environment_activation_command,
)
from test.test_runtime_resources import RESOURCE_CLUSTER_CONFIG


@pytest.mark.parametrize(
    "function_execution_context",
    [RESOURCE_CLUSTER_CONFIG],
    indirect=True,
)
def test_generated_cluster_scripts_activate_cvmfs_python(
    function_execution_context,
):
    config = function_execution_context.config

    execution_script = format_qsub_execution_script(
        context=function_execution_context,
        command="python train/single_train.py",
    )
    build_script = format_qsub_build_script(
        config=config,
        git_branch="main",
        git_commit_hash="0123456789abcdef",
    )

    assert cvmfs_python_activation_command() in execution_script
    assert str(CVMFS_PYTHON_SETUP_PATH) not in build_script


def test_definition_uses_the_locked_project_venv():
    definition = Path("lfvddp.def").read_text()

    assert definition.count("from frame.python_environment import CVMFS_PYTHON_SETUP_PATH") == 3
    assert 'python -m venv --system-site-packages "$CONTAINER_PROJECT_ROOT/.venv"' in definition
    assert "python -m pip install --no-cache-dir --upgrade uv" in definition
    assert "python -m uv sync --locked --active" in definition
    assert "source /app/.venv/bin/activate" in definition


def test_python_environment_activation_orders_cvmfs_before_venv(tmp_path):
    command = python_environment_activation_command(tmp_path)

    assert command == (
        f"{cvmfs_python_activation_command()} && "
        f"source {tmp_path / '.venv' / 'bin' / 'activate'}"
    )
