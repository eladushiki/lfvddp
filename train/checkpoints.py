from logging import warning
from pathlib import Path
from typing import Any, Optional

import torch

from frame.context.execution_context import ExecutionContext
from frame.file_structure import TRAINING_OUTCOMES_DIR_NAME


TRAINING_CHECKPOINT_SUFFIX = "checkpoint.pt"
CHECKPOINTS_DIR_NAME = "checkpoints"


def checkpoint_filename(model_name: str) -> str:
    return f"{model_name}.{TRAINING_CHECKPOINT_SUFFIX}"


def _torch_load(file_path: Path) -> dict[str, Any]:
    try:
        return torch.load(file_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(file_path, map_location="cpu")


def _checkpoint_dir(context: ExecutionContext) -> Path:
    if context.continue_from is None:
        return context.training_outcomes_dir

    checkpoint_dir = Path(context.continue_from) / TRAINING_OUTCOMES_DIR_NAME / CHECKPOINTS_DIR_NAME
    if context.array_index is not None:
        checkpoint_dir = checkpoint_dir / f"array_{context.array_index}"
    return checkpoint_dir


def save_training_checkpoint(
    context: ExecutionContext,
    model_name: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    training_history: dict[str, Any],
) -> Path:
    checkpoint_dir = _checkpoint_dir(context)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_filename(model_name)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    torch.save({
        "model_name": model_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "training_history": training_history,
        "array_index": context.array_index,
        "run_hash": context.run_hash,
    }, temporary_path)
    temporary_path.replace(checkpoint_path)
    return checkpoint_path


def find_latest_training_checkpoint(
    context: ExecutionContext,
    model_name: str,
    warn_missing: bool = True,
) -> Optional[tuple[Path, dict[str, Any]]]:
    if not context.is_continue or context.continue_from is None:
        return None

    checkpoint_path = _checkpoint_dir(context) / checkpoint_filename(model_name)
    if not checkpoint_path.exists():
        if warn_missing:
            warning(f"Continuation checkpoint does not exist: {checkpoint_path}")
        return None

    checkpoint = _torch_load(checkpoint_path)
    if checkpoint.get("model_name") != model_name:
        raise RuntimeError(f"Checkpoint {checkpoint_path} belongs to {checkpoint.get('model_name')}, not {model_name}")

    checkpoint_array_index = checkpoint.get("array_index")
    if checkpoint_array_index != context.array_index:
        warning(
            f"Checkpoint {checkpoint_path} belongs to array index {checkpoint_array_index}, "
            f"not {context.array_index}"
        )
        return None

    return checkpoint_path, checkpoint
