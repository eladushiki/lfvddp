from json import JSONEncoder, dump, load
from pathlib import Path
from typing import List


def load_dict_from_json(file_path: Path) -> dict:
    with open(file_path, 'r') as file:
        return load(file)


def save_dict_to_json(dictionary: dict, file_path: Path):
    with open(file_path, 'w') as file:
        dump(dictionary, file, indent=4, cls=FallbackJSONEncoder)


class FallbackJSONEncoder(JSONEncoder):
    def default(self, o):
        try:
            return super().default(o)
        except TypeError:
            try:  # todo: this is not good, also - document the type of product (although it is implied)
                return o.__dict__
            except AttributeError:
                return str(o)


def read_text_file_lines(file_path: Path) -> List[str]:
    """
    Read the content of a text file and return it as a string.
    """
    with open(file_path, 'r') as file:
        return file.readlines()
