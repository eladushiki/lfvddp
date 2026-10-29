from json import JSONDecodeError, JSONEncoder, dump, load
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import List, Dict, Any
import yaml

from frame.file_structure import (
    CONFIG_FILE_EXTENSIONS,
    CONFIGS_DIR_NAME,
    JSON_FILE_EXTENSION,
    YAML_FILE_EXTENSIONS,
)

JSON_READ_ATTEMPTS = 3
JSON_READ_RETRY_DELAY_SECONDS = 0.05


def _config_search_root(config_path: Path) -> Path:
    """Prefer a run directory's staged configs over its other descendants."""
    staged_configs_directory = config_path / CONFIGS_DIR_NAME
    return (
        staged_configs_directory if staged_configs_directory.is_dir() else config_path
    )


def expand_config_paths(config_paths: list[Path]) -> list[Path]:
    """Replace config directories with their recursively discovered files."""
    expanded_paths = []
    for config_path in config_paths:
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration path does not exist or is inaccessible: {config_path}"
            )
        if config_path.is_dir():
            config_search_root = _config_search_root(config_path)
            expanded_paths.extend(
                path
                for path in sorted(config_search_root.rglob("*"))
                if path.is_file()
                and path.suffix.lower().removeprefix(".") in CONFIG_FILE_EXTENSIONS
            )
        elif config_path.is_file():
            expanded_paths.append(config_path)
        else:
            raise ValueError(
                f"Configuration path is neither a regular file nor a directory: "
                f"{config_path}"
            )
    return expanded_paths


def load_dict_from_json(file_path: Path) -> dict:
    # Old workers may still rewrite files in place. Reopen on a bounded retry;
    # never turn persistent corruption into an apparently absent run.
    for attempt in range(JSON_READ_ATTEMPTS):
        try:
            with open(file_path, "r") as file:
                return load(file)
        except JSONDecodeError as error:
            if attempt == JSON_READ_ATTEMPTS - 1:
                error.add_note(
                    f"Invalid JSON in {file_path} after {JSON_READ_ATTEMPTS} reads"
                )
                raise
            sleep(JSON_READ_RETRY_DELAY_SECONDS)


def load_dict_from_yaml(file_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(file_path, "r") as file:
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
        displayed_extension = f".{file_extension}" if file_extension else "<none>"
        supported_formats = ", ".join(
            f".{extension}" for extension in CONFIG_FILE_EXTENSIONS
        )
        raise ValueError(
            f"Unsupported configuration file format {displayed_extension} for "
            f"{file_path}. Supported formats: {supported_formats}"
        )


def load_config_params_from_paths(config_paths: List[Path]) -> Dict[str, Any]:
    """Shallow-merge configuration files in their supplied order."""
    config_params = {}
    for config_path in config_paths:
        config_params.update(load_config_file(config_path))
    return config_params


def save_dict_to_json(dictionary: dict, file_path: Path):
    """Publish a complete JSON document without exposing partial writes."""
    file_path = Path(file_path)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            dump(dictionary, file, indent=4, cls=FallbackJSONEncoder)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, file_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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
    with open(file_path, "r") as file:
        return file.readlines()
