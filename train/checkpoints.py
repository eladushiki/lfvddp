import json
from logging import warning
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import torch

from frame.context.execution_context import ExecutionContext
from frame.context.run_descriptor import run_descriptor_matches
from frame.file_structure import (
    CHECKPOINTS_DIR_NAME,
    CONTEXT_FILE_NAME,
    SINGLE_TRAIN_SCRIPT_NAME,
    TRAINING_CHECKPOINT_SUFFIX,
    TRAINING_OUTCOMES_DIR_NAME,
)


def checkpoint_filename(model_name: str) -> str:
    return f"{model_name}.{TRAINING_CHECKPOINT_SUFFIX}"


def checkpoint_metadata_path(checkpoint_path: Path) -> Path:
    """Return the legacy metadata sidecar path for a training checkpoint."""

    return checkpoint_path.with_name(checkpoint_path.name + ".metadata.json")


def _torch_load(file_path: Path) -> dict[str, Any]:
    try:
        return torch.load(file_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(file_path, map_location="cpu")


def load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    """Load checkpoint metadata, accepting legacy JSON sidecars."""

    checkpoint = _torch_load(checkpoint_path)
    metadata = checkpoint.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise RuntimeError(f"Checkpoint {checkpoint_path} has invalid metadata.")
        return metadata

    metadata_path = checkpoint_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no metadata.")
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Unable to read checkpoint metadata sidecar {metadata_path}: {error}"
        ) from error
    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Checkpoint metadata sidecar {metadata_path} must contain a JSON object."
        )
    return metadata


def _checkpoint_dir(context: ExecutionContext) -> Path:
    return context.training_outcomes_dir


def _legacy_continuation_checkpoint_path(
    context: ExecutionContext,
    model_name: str,
) -> Optional[Path]:
    if context.continue_from is None:
        return None

    checkpoint_dir = (
        Path(context.continue_from) / TRAINING_OUTCOMES_DIR_NAME / CHECKPOINTS_DIR_NAME
    )
    if context.array_index is not None:
        checkpoint_dir = checkpoint_dir / f"array_{context.array_index}"
    return checkpoint_dir / checkpoint_filename(model_name)


def _single_train_checkpoint_paths(
    context: ExecutionContext,
    model_name: str,
) -> Iterable[Path]:
    if context.continue_from is None:
        return

    continue_from = Path(context.continue_from)
    dirsafe_runtag = getattr(
        getattr(context, "config", None), "config__dirsafe_runtag", None
    )
    for child_context_path in continue_from.glob(f"*/{CONTEXT_FILE_NAME}"):
        child_context = ExecutionContext.load_from_run_dir(child_context_path.parent)
        if child_context.array_index != context.array_index:
            continue
        if not run_descriptor_matches(
            child_context.run_descriptor,
            entrypoint=SINGLE_TRAIN_SCRIPT_NAME,
            dirsafe_runtag=dirsafe_runtag,
        ):
            continue

        checkpoint_path = (
            child_context_path.parent
            / TRAINING_OUTCOMES_DIR_NAME
            / checkpoint_filename(model_name)
        )
        if checkpoint_path.exists():
            yield checkpoint_path


def _continuation_checkpoint_paths(
    context: ExecutionContext,
    model_name: str,
) -> Iterable[Path]:
    """Yield both legacy and per-training-run checkpoint locations."""

    legacy_path = _legacy_continuation_checkpoint_path(context, model_name)
    if legacy_path is not None and legacy_path.exists():
        yield legacy_path
    yield from _single_train_checkpoint_paths(context, model_name)


def save_training_checkpoint(
    context: ExecutionContext,
    model_name: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    training_history: dict[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
    best_model_state_dict: Optional[dict[str, Any]] = None,
    best_loss: Optional[float] = None,
    best_epoch: Optional[int] = None,
) -> Path:
    checkpoint_dir = _checkpoint_dir(context)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / checkpoint_filename(model_name)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            if optimizer is not None
            else None,
            "training_history": training_history,
            "best_model_state_dict": best_model_state_dict,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "array_index": context.array_index,
            "run_hash": context.run_hash,
            "metadata": dict(metadata) if metadata is not None else None,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)

    return checkpoint_path


def find_latest_training_checkpoint(
    context: ExecutionContext,
    model_name: str,
    warn_missing: bool = True,
) -> Optional[tuple[Path, dict[str, Any]]]:
    if not context.is_continue or context.continue_from is None:
        return None

    candidates = []
    for checkpoint_path in _continuation_checkpoint_paths(context, model_name):
        checkpoint = _torch_load(checkpoint_path)
        if checkpoint.get("model_name") != model_name:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} belongs to {checkpoint.get('model_name')}, not {model_name}"
            )

        checkpoint_array_index = checkpoint.get("array_index")
        if checkpoint_array_index != context.array_index:
            warning(
                f"Checkpoint {checkpoint_path} belongs to array index {checkpoint_array_index}, "
                f"not {context.array_index}"
            )
            continue

        candidates.append((checkpoint_path, checkpoint))

    if not candidates:
        if warn_missing:
            warning(
                f"Continuation checkpoint does not exist for {model_name} "
                f"and array index {context.array_index} in {context.continue_from}"
            )
        return None

    return max(
        candidates,
        key=lambda candidate: (
            int(candidate[1].get("epoch", -1)),
            candidate[0].stat().st_mtime,
        ),
    )
