from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath


# project hierarchy
PROJECT_ROOT = Path(__file__).parent.parent
def get_relpath_from_local_root(local_absolute_path: PurePath) -> PurePosixPath:
    return PurePosixPath(local_absolute_path.relative_to(PROJECT_ROOT))
def get_remote_equivalent_path(remote_root_path: PurePosixPath, local_absolute_path: Path):
    return remote_root_path / get_relpath_from_local_root(local_absolute_path)
def get_local_equivalent_path(remote_root_path: PurePosixPath, remote_absolute_path: PurePosixPath):
    return PROJECT_ROOT / PurePath(remote_absolute_path.relative_to(remote_root_path))

CONFIGS_DIR = PROJECT_ROOT / "configs"

# file title constants
CONTEXT_FILE_NAME = "context.json"
JOB_OUTPUT_FILE_NAME = "job_output.txt"
RESULTS_BRIEFING_FILE_NAME = "results_briefing.txt"

# NN training
TRIANING_OUTCOMES_DIR_NAME = "training_outcomes"
TRAINING_HISTORY_FILE_NAME = "history.h5"
SINGLE_TRAINING_RESULT_POSTFIX = "res"
SINGLE_TRAINING_RESULT_FILE_NAME = "train." + SINGLE_TRAINING_RESULT_POSTFIX
AGGREGATED_TRAINING_RESULTS_FILE_NAME = "aggregated_results.txt"
WEIGHTS_OUTPUT_FILE_NAME = "training_result.weights.h5"

def convert_win_path_to_wsl(path: PureWindowsPath) -> PurePosixPath:
        slashed_path = (str(path)).replace("\\", "/")
        return PurePosixPath(f"/mnt/{slashed_path[0].lower()}/{slashed_path[3:]}")
