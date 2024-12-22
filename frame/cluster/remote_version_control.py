from pathlib import Path
from frame.cluster.call_scripts import run_remote_python
from frame.config_handle import ExecutionContext
from frame.file_structure import get_remote_equivalent_path
from frame.git_tools import get_commit_hash
from train.train_config import ClusterConfig

IS_SAME_AS_COMMIT_PY_ABS_PATH = Path(__file__).parent.absolute() / "is_same_as_commit.py"


def is_same_version_as_remote(
        context: ExecutionContext,
    ) -> bool:
    """
    Call this from the user's machine before submitting a
    job to verify that the remote commit hash matches the
    local one.
    """
    if not isinstance((config := context.config), ClusterConfig):
        raise ValueError(f"Expected ClusterConfig, got {context.config.__class__.__name__}")
    
    local_commit_hash = get_commit_hash()

    remote_script = get_remote_equivalent_path(config.cluster__remote_repository_dir, IS_SAME_AS_COMMIT_PY_ABS_PATH)
    comparison_result = run_remote_python(
        context,
        remote_script,
        script_arguments=[local_commit_hash],
    )

    return comparison_result == "0"
