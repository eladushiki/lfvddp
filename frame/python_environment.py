"""Commands for the project Python environment."""

from pathlib import Path
import shlex
from typing import Optional, Union


CVMFS_PYTHON_SETUP_PATH = Path(
    "/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh"
)
PROJECT_VENV_DIRECTORY_NAME = ".venv"
DEFAULT_UV_CACHE_DIR = Path("/tmp/lfvddp-uv-cache")


def uv_cache_directory(cache_dir: Optional[Union[str, Path]]) -> Path:
    """Return the configured UV cache directory or the project default."""
    return Path(cache_dir) if cache_dir else DEFAULT_UV_CACHE_DIR


def uv_cache_directory_shell_literal(cache_dir: Optional[Union[str, Path]]) -> str:
    """Return a shell-safe literal for the configured UV cache directory."""
    return shlex.quote(str(uv_cache_directory(cache_dir)))


def uv_cache_directory_export_command(cache_dir: Optional[Union[str, Path]]) -> str:
    """Return the shell command that configures UV's cache directory."""
    return f"export UV_CACHE_DIR={uv_cache_directory_shell_literal(cache_dir)}"


def singularity_uv_cache_directory_export_command(
    cache_dir: Optional[Union[str, Path]],
) -> str:
    """Return the shell command that forwards the UV cache into a container."""
    return "export SINGULARITYENV_UV_CACHE_DIR=" + shlex.quote(
        str(uv_cache_directory(cache_dir))
    )


def cvmfs_python_activation_command() -> str:
    """Return the shell command that selects the project Python interpreter."""
    return f"source {shlex.quote(str(CVMFS_PYTHON_SETUP_PATH))}"


def virtual_environment_activation_command(project_root: Path) -> str:
    """Return the shell command that activates the project's virtual environment."""
    venv_activate = project_root / PROJECT_VENV_DIRECTORY_NAME / "bin" / "activate"
    return f"source {shlex.quote(str(venv_activate))}"


def python_environment_activation_command(project_root: Path) -> str:
    """Return commands that activate CVMFS Python and its project venv in order."""
    return " && ".join(
        (
            cvmfs_python_activation_command(),
            virtual_environment_activation_command(project_root),
        )
    )
