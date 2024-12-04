from os.path import relpath
from pathlib import Path
from frame.cluster.call_scripts import run_remote_python
from frame.file_structure import PROJECT_ROOT
from frame.git_tools import get_commit_hash
from train.train_config import ClusterConfig


IS_SAME_AS_COMMIT_PY_PATH = Path(__file__).parent / "is_same_as_commit.py"
IS_SAME_AS_COMMIT_PY_RELPATH_FROM_ROOT = Path(relpath(PROJECT_ROOT, IS_SAME_AS_COMMIT_PY_PATH))

def is_same_version_as_remote(
        config: ClusterConfig,

    ) -> bool:
    """
    Call this from the user's machine before submitting a
    job to verify that the remote commit hash matches the
    local one.
    """
    local_commit_hash = get_commit_hash()

    remote_script = config.cluster__remote_repository_dir / IS_SAME_AS_COMMIT_PY_RELPATH_FROM_ROOT
    comparison_result = run_remote_python(
        config,
        config.cluster__remote_repository_dir,
        python_script = remote_script,
        script_arguments=[local_commit_hash]
    )

    return comparison_result == "0"
