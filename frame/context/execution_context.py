from contextlib import contextmanager
import random
from numpy import random as npramdom
from frame.config_handle import Config
from frame.file_storage import save_dict_to_json
from frame.file_structure import CONTEXT_FILE_NAME
from frame.context.execution_products import ExecutionProducts
from frame.git_tools import get_commit_hash, is_git_head_clean
from frame.time_tools import get_time_and_date_string, get_unix_timestamp


from dataclasses import dataclass, field
from os import getpid, makedirs
from pathlib import Path
from sys import argv
from typing import Any, List


@dataclass
class ExecutionContext:
    commit_hash: str
    config: Config
    command_line_args: List[str]
    time: str = get_time_and_date_string()
    random_seed: int = get_unix_timestamp()
    is_debug_mode: bool = False
    run_successful: bool = False
    _products: ExecutionProducts = field(default=ExecutionProducts(), init=False)

    def __post_init__(self):
        # Initialize once unique output directory
        makedirs(self.unique_out_dir, exist_ok=False)

    @property
    def _unique_descriptor(self) -> str:
        running_file = argv[0].split('/')[-1]
        process_id = getpid()
        return f"run_at_{self.time}_of_{running_file}_on_commit_{self.commit_hash[:5]}_pid_{process_id}"

    @property
    def unique_out_dir(self) -> Path:
        return self.config.out_dir / self._unique_descriptor

    def document_created_product(self, product_descriptor: Any):
        self._products.add_product(product_descriptor)

    @staticmethod
    def serialize(object) -> dict:
        series = object.__dict__.copy()

        # Convert non-serializable objects
        for key, value in series.items():
            if isinstance(value, Path):
                series[key] = str(value)
            elif isinstance(value, Config):
                series[key] = ExecutionContext.serialize(value)

        return series

    def save_to_out_file(self) -> None:
        save_dict_to_json(ExecutionContext.serialize(self), self.unique_out_dir / CONTEXT_FILE_NAME)


@contextmanager
def version_controlled_execution_context(config: Config, command_line_args: List[str], is_debug_mode: bool = False):
    """
    Create a context which should contain any run dependent information.
    The data is later stored in the output_path for documentatino.
    """
    # Force run on strict commit
    if not is_debug_mode and not is_git_head_clean():
        raise RuntimeError("Commit changes before running the script.")

    # Initialize
    context = ExecutionContext(get_commit_hash(), config, command_line_args, is_debug_mode=is_debug_mode)
    random.seed(context.random_seed)
    npramdom.seed(context.random_seed)

    # Save in case run terminates prematurely
    context.save_to_out_file()

    # Do everyting, add imoprttant stufff as parameters to context object
    yield context

    # Overwrite saved context at end of run
    context.run_successful = True
    context.save_to_out_file()
