from typing import Any, Mapping

from frame.context.execution_context import ExecutionContext
from frame.file_structure import TENSORBOARD_LOG_DIR_NAME
from frame.file_system.training_history import HistoryKeys


def log_t_history_to_tensorboard(
    context: ExecutionContext,
    sample_name: str,
    history: Mapping[str, Any],
) -> None:
    """Log the numerator, denominator, and derived t progress for one sample."""
    from torch.utils.tensorboard import SummaryWriter

    log_dir = context.training_outcomes_dir / TENSORBOARD_LOG_DIR_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        log_t_history(writer, sample_name, history)
    finally:
        writer.close()


def log_t_history(writer, sample_name: str, history: Mapping[str, Any]) -> None:
    """Write an already-derived t history to a TensorBoard-compatible writer."""
    epochs = history[HistoryKeys.EPOCH.value]
    for key in (
        HistoryKeys.NUMERATOR.value,
        HistoryKeys.DENOMINATOR.value,
        HistoryKeys.T.value,
    ):
        for epoch, value in zip(epochs, history[key]):
            writer.add_scalar(f"{sample_name}/{key}", float(value), int(epoch))
