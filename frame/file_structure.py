from pathlib import Path, PurePath, PurePosixPath


# project hierarchy
PROJECT_ROOT = Path(__file__).parent.parent
def get_relpath_from_local_root(local_absolute_path: PurePath) -> PurePosixPath:
    return PurePosixPath(local_absolute_path.relative_to(PROJECT_ROOT))


# File extensions
## Textual data
JSON_FILE_EXTENSION = "json"
TEXT_FILE_EXTENSION = "txt"
## Training logs
TRAINING_HISTORY_FILE_EXTENSION = "history_h5"
TRAINING_WEIGHTS_FILE_EXTENSION = "weights_h5"
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
WEIGHTS_OUTPUT_FILE_NAME = f"training_result.{TRAINING_WEIGHTS_FILE_EXTENSION}"
