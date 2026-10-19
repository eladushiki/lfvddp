"""Embed a fitted null as a feasible candidate in an adaptive alternative."""

import logging
from typing import TYPE_CHECKING

import torch

from data_tools.data_utils import ShiftAndNormalizationFactor
from neural_networks.function_spaces import AdaptiveNeuralFunction

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from neural_networks.differentiating_model import (
        DifferentiatingModel,
        _PreparedTrainingData,
    )


def validate_null_initialization_checkpoint(checkpoint, enabled: bool) -> None:
    saved = checkpoint.get("initialize_from_null", False)
    if type(saved) is not bool or saved != enabled:
        raise ValueError(
            "Continuation train__initialize_from_null differs from checkpoint."
        )


def prepare_null_normalization(
    model: "DifferentiatingModel", null_model: "DifferentiatingModel | None"
) -> None:
    if null_model is None or null_model._is_numerator:
        raise ValueError(
            "Null-initialized alternative requires a fitted denominator model."
        )
    if not model._is_numerator:
        raise ValueError("Only an alternative can be initialized from the null.")
    if (
        null_model._function_space_config.nuisance
        != model._function_space_config.nuisance
    ):
        raise ValueError("Null and alternative nuisance specifications differ.")
    if null_model._dtype != model._dtype:
        raise ValueError("Null and alternative dtypes differ.")
    if null_model.nuisance_function_space is not None and null_model._best_loss is None:
        raise ValueError(
            "The learned denominator must be fitted before null initialization."
        )
    if null_model._norm_factor is None:
        raise ValueError(
            "The denominator must be fitted before initializing the alternative."
        )
    model._restore_checkpoint_normalization_if_requested()
    if model._norm_factor is None:
        model._norm_factor = ShiftAndNormalizationFactor.from_mapping(
            null_model._norm_factor.to_mapping()
        )
    if model._norm_factor.to_mapping() != null_model._norm_factor.to_mapping():
        raise ValueError(
            "Null and alternative normalization maps differ during continuation."
        )


def retain_null_candidate(
    model: "DifferentiatingModel",
    null_model: "DifferentiatingModel",
    data: "_PreparedTrainingData",
    *,
    initialize: bool,
) -> bool:
    """Copy tensors, evaluate outside the budget, and preserve live resumed parameters."""
    signal = model.signal_region_shift_network
    if not isinstance(signal, AdaptiveNeuralFunction):
        raise TypeError(
            "Null initialization requires an adaptive_neural signal function."
        )
    live_state = None if initialize else model._model_state_snapshot()
    previous_best = model._best_loss
    try:
        if model.nuisance_function_space is not None:
            if null_model.nuisance_function_space is None:
                raise ValueError("Null and alternative nuisance spaces differ.")
            null_state = null_model._best_model_state_dict
            if null_state is None:
                null_state = null_model._model_state_snapshot()
            prefix = "nuisance_function_space."
            nuisance_state = {
                name.removeprefix(prefix): value
                for name, value in null_state.items()
                if name.startswith(prefix)
            }
            model.nuisance_function_space.load_state_dict(nuisance_state, strict=True)
        elif null_model.nuisance_function_space is not None:
            raise ValueError("Null and alternative nuisance spaces differ.")
        with torch.no_grad():
            signal.output.weight.zero_()
            signal.output.bias.zero_()
            loss = model(data=data)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                "The fitted null has a nonfinite feasible alternative loss."
            )
        logger.info(
            "Null candidate for %s: loss=%.17g (unbudgeted, no gradients; fresh=%s).",
            model._name,
            float(loss.detach().cpu()),
            initialize,
        )
        model._record_best_model_state(loss, -1, model._model_state_snapshot())
    finally:
        if live_state is not None:
            model.load_state_dict(live_state, strict=True)
    return previous_best != model._best_loss
