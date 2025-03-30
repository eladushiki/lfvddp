from json import JSONEncoder, dump, load
from pathlib import Path


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
