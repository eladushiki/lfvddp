from pathlib import Path, PurePath, PurePosixPath


# project hierarchy
PROJECT_NAME = "lfvddp"
LOCAL_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CONFIGS_DIR = LOCAL_PROJECT_ROOT / "configs"
TRAIN_DIR = LOCAL_PROJECT_ROOT / "train"
PLOT_DIR = LOCAL_PROJECT_ROOT / "plot"
SINGULARITY_DEFINITION_FILE = LOCAL_PROJECT_ROOT / f"{PROJECT_NAME}.def"

def get_relpath_from_local_root(local_absolute_path: PurePath) -> PurePosixPath:
    return PurePosixPath(local_absolute_path.relative_to(LOCAL_PROJECT_ROOT))


# At containsr
CONTAINER_PROJECT_ROOT = PurePosixPath("/app")

def path_as_in_container(local_path: Path) -> PurePosixPath:
    relative_path = get_relpath_from_local_root(local_path)
    return CONTAINER_PROJECT_ROOT / relative_path


# File extensions
## Textual data
JSON_FILE_EXTENSION = "json"
YAML_FILE_EXTENSION = "yaml"
YML_FILE_EXTENSION = "yml"
YAML_FILE_EXTENSIONS = [YAML_FILE_EXTENSION, YML_FILE_EXTENSION]
TEXT_FILE_EXTENSION = "txt"
BASH_FILE_EXTENSION = "sh"
## Training logs
TRAINING_LOG_FILE_EXTENSION = "h5"
TRAINING_HISTORY_LOG_FILE_SUFFIX = "history." + TRAINING_LOG_FILE_EXTENSION
TRAINING_WEIGHTS_LOG_FILE_SUFFIX = "weights." + TRAINING_LOG_FILE_EXTENSION
SINGLE_TRAINING_RESULT_FILE_EXTENSION = TEXT_FILE_EXTENSION
## Plotting
PLOT_FILE_EXTENSION = "png"

# File title constants
CONTEXT_FILE_NAME = f"context.{JSON_FILE_EXTENSION}"
JOB_OUTPUT_FILE_NAME = f"job_output.{TEXT_FILE_EXTENSION}"
RESULTS_BRIEFING_FILE_NAME = f"results_briefing.{TEXT_FILE_EXTENSION}"

# NN training
TRAINING_OUTCOMES_DIR_NAME = "training_outcomes"
SINGLE_TRAINING_RESULT_FILE_NAME = f"final_t.{SINGLE_TRAINING_RESULT_FILE_EXTENSION}"
WEIGHTS_OUTPUT_FILE_NAME = f"training_result.{TRAINING_WEIGHTS_LOG_FILE_SUFFIX}"
TENSORBOARD_LOG_DIR_NAME = f"tensorboard_logs"
