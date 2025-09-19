import numpy as np

from data_tools.data_utils import DataSet
from frame.context.execution_context import ExecutionContext
from frame.file_structure import TRAINING_HISTORY_LOG_FILE_SUFFIX, WEIGHTS_OUTPUT_FILE_NAME
from neural_networks.NPLM.src.NPLM.NNutils import imperfect_model, np
from tensorflow.keras.models import Model # type: ignore
from train.train_config import TrainConfig


import os
from typing import Any, Dict


def save_training_outcomes(
        context: ExecutionContext,
        model_history: Dict[str, Any],
        tau_model: imperfect_model,
    ) -> None:
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    ## Training log
    os.makedirs(context.training_outcomes_dir, exist_ok=True)

    # Save training
    context.save_and_document_model_history(model_history, context.training_outcomes_dir / f"{tau_model.name}.{TRAINING_HISTORY_LOG_FILE_SUFFIX}")
    context.save_and_document_model_weights(tau_model, context.training_outcomes_dir / f"{tau_model.name}_{WEIGHTS_OUTPUT_FILE_NAME}")


def predict_sample_ndf_hypothesis_weights(trained_model: Model, predicted_distribution_corrected_size: float, reference_ndf_estimation: DataSet) -> np.ndarray:
    model_prediction = trained_model.predict(reference_ndf_estimation)[:, 0]  # Corresponds the 1 dimension of array output
    hypothesis_weights = np.expand_dims(np.exp(model_prediction), axis=1) * reference_ndf_estimation.histogram_weight_mask
    return predicted_distribution_corrected_size / reference_ndf_estimation.corrected_n_samples * hypothesis_weights
