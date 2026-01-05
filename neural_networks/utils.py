import numpy as np
import torch.nn as nn

from data_tools.data_utils import DataSet
from frame.context.execution_context import ExecutionContext
from frame.file_structure import TRAINING_HISTORY_LOG_FILE_SUFFIX, TENSORBOARD_LOG_DIR_NAME, WEIGHTS_OUTPUT_FILE_NAME
from neural_networks.NPLM.src.NPLM.NNutils import imperfect_model, np
from train.train_config import TrainConfig


import os
from typing import Any, Dict, Protocol
from pathlib import Path


class ContextedModel(Protocol):
    """
    Common interface for neural network models (PyTorch and Keras-based).
    Defines the minimal contract that all models in this framework must fulfill.
    """
    _name: str
    
    def predict(self, data: DataSet) -> np.ndarray:
        """Make predictions on a DataSet."""
        ...
    
    def save_weights(self, file_path: Path) -> None:
        """Save model weights to a file."""
        ...


MAX_PREDICTION_CUTOFF = 20.0
MIN_PREDICTION_CUTOFF = -75.0


def get_model_logging_dir(context: ExecutionContext, model_name: str) -> Path:
    """Get the logging directory for a model."""
    return context.training_outcomes_dir / TENSORBOARD_LOG_DIR_NAME / model_name


def save_training_outcomes(
        context: ExecutionContext,
        model_history: Dict[str, Any],
        tau_model: ContextedModel,
    ) -> None:
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    ## Training log
    model_output_dir = get_model_logging_dir(context, tau_model._name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Save training
    context.save_and_document_model_history(model_history, model_output_dir / f"{tau_model._name}.{TRAINING_HISTORY_LOG_FILE_SUFFIX}")
    context.save_and_document_model_weights(tau_model, model_output_dir / f"{tau_model._name}_{WEIGHTS_OUTPUT_FILE_NAME}")


def predict_sample_ndf_hypothesis_weights(trained_model: ContextedModel, predicted_distribution_corrected_size: float, reference_ndf_estimation: DataSet) -> np.ndarray:
    model_prediction = trained_model.predict(data=reference_ndf_estimation)
    hypothesis_weights = np.expand_dims(np.exp(model_prediction), axis=1) * reference_ndf_estimation.histogram_weight_mask
    return predicted_distribution_corrected_size / reference_ndf_estimation.corrected_n_samples * hypothesis_weights
