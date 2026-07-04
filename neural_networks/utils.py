import numpy as np

from data_tools.data_utils import DataSet
from frame.context.execution_context import ExecutionContext
from frame.file_structure import TRAINING_HISTORY_LOG_FILE_SUFFIX, TENSORBOARD_LOG_DIR_NAME, WEIGHTS_OUTPUT_FILE_NAME
from train.train_config import TrainConfig


import os
from abc import abstractmethod
from typing import Any, Dict, Protocol
from pathlib import Path


class ContextedModel(Protocol):
    """
    Common interface for neural network models (PyTorch and Keras-based).
    Defines the minimal contract that all models in this framework must fulfill.
    """
    _name: str

    @abstractmethod
    def save_parameters(self, file_path: Path) -> None:
        """Save model weights to a file."""
        ...


def get_model_logging_dir(context: ExecutionContext, model_name: str) -> Path:
    """Get the logging directory for a model."""
    return context.training_outcomes_dir


def save_training_outcomes(
        context: ExecutionContext,
        model_history: Dict[str, Any],
        tau_model: ContextedModel,
    ) -> None:
    save_training_history_outcome(context, model_history, tau_model._name)
    model_output_dir = get_model_logging_dir(context, tau_model._name)
    context.save_and_document_model_parameters(tau_model, model_output_dir / f"{tau_model._name}_{WEIGHTS_OUTPUT_FILE_NAME}")


def save_training_history_outcome(
        context: ExecutionContext,
        model_history: Dict[str, Any],
        model_name: str,
    ) -> None:
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    ## Training log
    model_output_dir = get_model_logging_dir(context, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Save training
    context.save_and_document_model_history(model_history, model_output_dir / f"{model_name}.{TRAINING_HISTORY_LOG_FILE_SUFFIX}")


def prediction_to_sample_ndf_hypothesis_weights(
    model_prediction: np.ndarray,
    predicted_distribution_corrected_size: float,
    reference_ndf_estimation: DataSet,
) -> np.ndarray:
    model_prediction = np.asarray(model_prediction).reshape(-1)
    hypothesis_weights = np.exp(model_prediction)[:, None] * reference_ndf_estimation.histogram_weight_mask
    return (
        predicted_distribution_corrected_size
        / reference_ndf_estimation.corrected_n_samples
        * hypothesis_weights
    )
