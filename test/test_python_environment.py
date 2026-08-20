from pathlib import Path

import pytest

from frame.command_line.execution import (
    format_qsub_build_script,
    format_qsub_execution_script,
)
from frame.python_environment import (
    CVMFS_PYTHON_SETUP_PATH,
    DEFAULT_UV_CACHE_DIR,
    cvmfs_python_activation_command,
    python_environment_activation_command,
    singularity_uv_cache_directory_export_command,
    uv_cache_directory_export_command,
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
    assert uv_cache_directory_export_command(None) in execution_script
    assert singularity_uv_cache_directory_export_command(None) in execution_script
    assert str(CVMFS_PYTHON_SETUP_PATH) not in build_script
    assert f"export UV_CACHE_DIR={DEFAULT_UV_CACHE_DIR}" in build_script


def test_definition_activates_the_bound_project_venv_at_runtime():
    definition = Path("lfvddp.def").read_text()

    assert definition.count("from frame.python_environment import CVMFS_PYTHON_SETUP_PATH") == 1
    assert "python -m uv sync --locked --active" not in definition
    assert "if [ -x /app/.venv/bin/python ]; then" in definition
    assert "source /app/.venv/bin/activate" not in definition
    assert "export VIRTUAL_ENV=/app/.venv" in definition
    assert 'export PATH="$VIRTUAL_ENV/bin:$PATH"' in definition
    assert "unset PYTHONHOME" in definition
    assert "test -f /app/uv.lock" in definition


def test_python_environment_activation_orders_cvmfs_before_venv(tmp_path):
    command = python_environment_activation_command(tmp_path)

    assert command == (
        f"{cvmfs_python_activation_command()} && "
        f"source {tmp_path / '.venv' / 'bin' / 'activate'}"
    )
