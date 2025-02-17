from argparse import ArgumentParser
from glob import glob
from logging import warning
from pathlib import Path
from typing import List
from frame.file_structure import AGGREGATED_TRAINING_RESULTS_FILE_NAME, SINGLE_TRAINING_RESULT_POSTFIX


def aggregate_train_results(parent_directory: Path) -> List[float]:
    if not parent_directory.is_dir():
        raise NotADirectoryError(f"Parent directory {parent_directory} does not exist")
    
    files_in_output_dir = glob(str(parent_directory) + f"/**/*.{SINGLE_TRAINING_RESULT_POSTFIX}", recursive=True)
    
    # Append all legal files
    aggregated_results = []
    for file in files_in_output_dir:
        try:
            with open(file, 'r') as f:
                content = f.read()
            result = float(content)
        except ValueError:
            warning(f"Could not parse training result from file {file}")
            continue
        aggregated_results.append(result)

    return aggregated_results


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("parent_directory", type=Path)
    args = parser.parse_args()

    train_results = aggregate_train_results(args.parent_directory)

    # todo: this also should be run in a context, yet requires different args
    
    with open(args.parent_directory / AGGREGATED_TRAINING_RESULTS_FILE_NAME, 'w') as f:
        f.writelines([f"{num}\n" for num in train_results])
