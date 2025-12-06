from json import JSONEncoder, dump, load
from pathlib import Path
from typing import List, Dict, Any
import yaml

from frame.file_structure import JSON_FILE_EXTENSION, YAML_FILE_EXTENSIONS


def load_dict_from_json(file_path: Path) -> dict:
    with open(file_path, 'r') as file:
        return load(file)


def load_dict_from_yaml(file_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def load_config_file(file_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file based on file extension.
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        ValueError: If file format is not supported
    """
    file_extension = file_path.suffix.lower()[1:]  # Remove the leading dot
    
    if file_extension == JSON_FILE_EXTENSION:
        return load_dict_from_json(file_path)
    elif file_extension in YAML_FILE_EXTENSIONS:
        return load_dict_from_yaml(file_path)
    else:
        raise ValueError(f"Unsupported configuration file format: {file_extension}. Supported formats: .json, .yaml, .yml")


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
