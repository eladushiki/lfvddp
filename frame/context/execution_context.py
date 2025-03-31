from contextlib import contextmanager
from logging import info
import random
from numpy import random as npramdom
from matplotlib.figure import Figure
from frame.config_handle import UserConfig
from frame.file_system.image_storage import save_figure
from frame.file_system.textual_data import save_dict_to_json
from frame.file_structure import CONTEXT_FILE_NAME, TRIANING_OUTCOMES_DIR_NAME
from frame.context.execution_products import ExecutionProducts
from frame.git_tools import get_commit_hash, is_git_head_clean
from frame.time_tools import get_time_and_date_string, get_unix_timestamp
from tensorflow.keras.models import Model


from dataclasses import dataclass, field
from os import getpid, makedirs, sep
from pathlib import Path
from sys import argv
from typing import Any, List


@dataclass
class ExecutionContext:
    commit_hash: str
    config: UserConfig
    command_line_args: List[str]
    time: str = get_time_and_date_string()
    random_seed: int = get_unix_timestamp() + getpid()
    is_debug_mode: bool = False
    run_successful: bool = False
    products: ExecutionProducts = field(default=ExecutionProducts())

    def __post_init__(self):
        # Initialize once unique output directory
        makedirs(self.unique_out_dir, exist_ok=False)
        random.seed(self.random_seed)
        npramdom.seed(self.random_seed)

    @property
    def _unique_descriptor(self) -> str:
        running_file = argv[0].split(sep)[-1]
        process_id = getpid()
        return f"run_at_{self.time}_of_{running_file}_on_commit_{self.commit_hash[:5]}_pid_{process_id}"

    @property
    def unique_out_dir(self) -> Path:
        return Path(self.config.config__out_dir) / self._unique_descriptor

    @property
    def training_outcomes_dir(self) -> Path:
        return self.unique_out_dir / TRIANING_OUTCOMES_DIR_NAME

    def document_created_product(self, product_descriptor: Any):
        self.products.add_product(product_descriptor)
        info(f"Documented product: {product_descriptor}")

    @staticmethod
    def serialize(object) -> dict:
        series = object.__dict__.copy()

        # Convert non-serializable objects
        for key, value in series.items():
            if isinstance(value, Path):
                series[key] = str(value)
            elif isinstance(value, UserConfig):
                series[key] = ExecutionContext.serialize(value)

        return series

    # todo: export to decorator and add os.makedirs(out_dir, exist_ok=False)
    def save_and_document_dict(self, dict: dict, file_path: Path):
        save_dict_to_json(dict, file_path)
        self.document_created_product(file_path)

    def save_and_document_figure(self, figure: Figure, path: Path):
        save_figure(figure, path)
        self.document_created_product(path)

    def save_and_document_text(self, text: str, path: Path):
        with open(path, 'w') as file:
            file.write(text)
        self.document_created_product(path)

    def save_and_document_model_weights(self, model: Model, path: Path):
        model.save_weights(path)
        self.document_created_product(path)

    def close(self):
        self.run_successful = True
        self.save_self_to_out_file()

    def save_self_to_out_file(self) -> None:
        save_dict_to_json(ExecutionContext.serialize(self), self.unique_out_dir / CONTEXT_FILE_NAME)


@contextmanager
def version_controlled_execution_context(config: UserConfig, command_line_args: List[str], is_debug_mode: bool = False):
    """
    Create a context which should contain any run dependent information.
    The data is later stored in the output_path for documentatino.
    """
    # Force run on strict commit
    if not is_debug_mode and not is_git_head_clean():
        raise RuntimeError("Commit changes before running the script.")

    # Initialize
    context = ExecutionContext(get_commit_hash(), config, command_line_args, is_debug_mode=is_debug_mode)

    # Save in case run terminates prematurely
    context.save_self_to_out_file()

    # Do everyting, add imoprttant stufff as parameters to context object
    yield context

    # Overwrite saved context at end of run
    context.close()
