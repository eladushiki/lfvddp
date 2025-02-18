from json import dump, load
from pathlib import Path


def load_dict_from_json(file_path: Path) -> dict:
    with open(file_path, 'r') as file:
        return load(file)


def save_dict_to_json(dictionary: dict, file_path: Path) -> None:
    with open(file_path, 'w') as file:
        dump(dictionary, file, indent=4)
