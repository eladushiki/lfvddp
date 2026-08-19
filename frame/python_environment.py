"""Commands for the project Python environment."""

from pathlib import Path
import shlex


CVMFS_PYTHON_SETUP_PATH = Path(
    "/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh"
)
PROJECT_VENV_DIRECTORY_NAME = ".venv"


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
