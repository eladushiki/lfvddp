from argparse import ArgumentParser
from glob import glob
from logging import warning
from pathlib import Path
from typing import List, Tuple
from csv import writer as csv_writer
from frame.file_structure import AGGREGATED_TRAINING_RESULTS_FILE_NAME, SINGLE_TRAINING_RESULT_FILE_EXTENSION


def aggregate_train_results(parent_directory: Path) -> List[Tuple[float, str]]:
    if not parent_directory.is_dir():
        raise NotADirectoryError(f"Parent directory {parent_directory} does not exist")
    
    files_in_output_dir = glob(str(parent_directory) + f"/**/*.{SINGLE_TRAINING_RESULT_FILE_EXTENSION}", recursive=True)
    
    # Append all legal files
    aggregated_results = []
    for file in files_in_output_dir:
        try:
            with open(file, 'r') as f:
                content = f.read()
            result = (float(content), file)
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

    with open(args.parent_directory / AGGREGATED_TRAINING_RESULTS_FILE_NAME, 'w', newline='\n') as csvfile:
        csvwriter = csv_writer(csvfile)
        # csvwriter.writerow(['Result', 'File'])  # todo: wrap with common function to ensure headers are ok
        csvwriter.writerows(train_results)
