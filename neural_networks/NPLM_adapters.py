from logging import info
from time import time
from typing import Tuple
from data_tools.data_utils import DataSet
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_system.training_history import HistoryKeys
from neural_networks.NPLM.src.NPLM.NNutils import imperfect_loss, imperfect_model, np, train_model
import numpy as np
from tensorflow.keras import optimizers # type: ignore
from tensorflow.keras.models import Model # type: ignore

from neural_networks.utils import save_training_outcomes
from neural_networks.weights.taylor_expansion_net.parameters import parNN_list
from train.train_config import TrainConfig


def build_NPLM_shape_dictionary_list():
    # todo: this should be drawn from the config
    return [parNN_list['scale']]  # todo: this should be of the length of deltas? Look @ imperfect_model implementation


def get_NPLM_model(
        config: TrainConfig,
        is_tau: bool = True,  # else, delta
        name: str = "tau_model",
    ) -> imperfect_model:
    """
    Generate an NPLM imperfect model according to our configuration
    """
    # Solely for type hinting to take place
    if not isinstance(config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    
    if config.train__nuisance_correction_types == "" and not is_tau:
        raise ValueError("No Delta term needed when training without nuisances")

    ## Treating nuisance parameters
    # normalization of the nuisance parameters, $\nu_n$ in the text.
    # Only intact if correction type is "NORM" or "SHAPE"
    SIGMA_N   = config.train__norm_nuisance_std
    NU_N      = config.train__norm_nuisance_mean
    NUR_N     = config.train__norm_nuisance_reference
    NU0_N     = np.random.normal(loc=NU_N, scale=SIGMA_N, size=1)[0]

    # shape of the nuisance parameters, $\nu_s$ in the text
    # Only intact if correction type is "SHAPE"
    SIGMA_S   = np.array([config.train__shape_nuisance_std])
    NU_S      = np.array([config.train__shape_nuisance_mean])
    NUR_S     = np.array([config.train__shape_nuisance_reference])
    NU0_S     = np.random.normal(loc=NU_S[0], scale=SIGMA_S[0], size=1)[0]

    # Get Tau term model
    tau_model = imperfect_model(
        name=name,
        input_shape=(None, config.train__nn_input_dimension),
        NU_S=NU_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S,  # Lists of parameters for nuisance initial values
        NU_N=NU_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
        correction = config.train__nuisance_correction_types,  # Which nuisance to compensate for
        shape_dictionary_list = build_NPLM_shape_dictionary_list(),  # This is used in "SHAPE" correction case
        BSMarchitecture = config.train__nn_architecture,
        BSMweight_clipping = config.train__nn_weight_clipping,
        train_f = is_tau,  # = Should create model.BSMfinderNet = is training also for Tau (else, just Delta as in NPLM paper). We generally want to train for both.
        train_nu = config.train__data_is_train_for_nuisances,   # Should the nuisances change or stick with initial values
    )
    tau_model.summary(print_fn=lambda x: info(x))  # Otherwise, model.summary() just uses print()

    # Nuisance unused parameters set, later causes train to access unintialized members in these cases:
    if config.train__nuisance_correction_types != "SHAPE":
        tau_model.nu_s = 0
    if config.train__nuisance_correction_types == "":
        tau_model.nu_n = 0

    return tau_model


def build_feature_for_NPLM_model_train(exp_dataset, aux_dataset):
    return exp_dataset + aux_dataset


def build_target_for_NPLM_model_loss(sample_dataset: DataSet, reference_dataset: DataSet):
    # Is sample boolean mask
    _ones_like_sample = np.ones(shape=(sample_dataset.n_samples,))  # 1 for dim 1 because the NN's output is 1D.
    _zeros_like_reference = np.zeros(shape=(reference_dataset.n_samples,))
    _is_sample_mask = np.concatenate((_ones_like_sample, _zeros_like_reference), axis=0)

    # Weight mask, multiplies loss
    _sample_weights = sample_dataset._weight_mask
    _reference_weights = reference_dataset._weight_mask * sample_dataset.corrected_n_samples * 1. / reference_dataset.corrected_n_samples
    _weight_mask = np.concatenate((_sample_weights, _reference_weights), axis=0)

    # NPLM's format
    _is_sample_mask_expanded = np.expand_dims(_is_sample_mask, axis=1)
    _weight_mask_expanded = np.expand_dims(_weight_mask, axis=1)
    loss_mask = np.concatenate((_is_sample_mask_expanded, _weight_mask_expanded), axis=1)

    return loss_mask


def train_NPML_model(
        context: ExecutionContext,
        model: imperfect_model,
        sample_dataset: DataSet,
        reference_dataset: DataSet
    ) -> float:
    """
    returns:
        The final loss according to the model
    """
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    
    ## Done preparing sample
    feature_dataset = build_feature_for_NPLM_model_train(sample_dataset, reference_dataset)
    target_structure = build_target_for_NPLM_model_loss(sample_dataset, reference_dataset)

    # Train
    info("Starting training")
    t0 = time()
    
    # Train either of the nuisance parameters, or
    tau_model_history = train_model(
        model=model,
        feature=np.array(feature_dataset.events, dtype=np.float32),
        target=np.array(target_structure, dtype=np.float32),
        loss=imperfect_loss,  # This is (11) in "Learning New Physics from a Machine", D'Angolo et al.
        optimizer=optimizers.legacy.Adam(),
        total_epochs=config.train__epochs,
        patience=config.train__number_of_epochs_for_checkpoint,
        clipping=config.train__nn_weight_clipping > 0,
        verbose=False,
    )
    tau_history = np.array(tau_model_history[HistoryKeys.LOSS.value])                
    
    info(f'Training time (seconds): {time() - t0}')

    final_loss = calc_t_test_statistic(tau_history[-1])
    info(f'Observed t test statistic: {final_loss}')
    
    save_training_outcomes(
        context,
        model_history=tau_model_history,
        tau_model=model,
    )

    return final_loss


def calc_t_NPLM(
        context: ExecutionContext,
        sample_dataset: DataSet,
        reference_dataset: DataSet,
        name: str,
    ) -> Tuple[Model, float]:
    tau_model = get_NPLM_model(
        context.config,
        name=name + "_tau",
    )
    tau = train_NPML_model(
        context=context,
        model=tau_model,
        sample_dataset=sample_dataset,
        reference_dataset=reference_dataset,
    )

    if context.config.train__nuisance_correction_types != "":
        delta_model = get_NPLM_model(
            context.config,
            is_tau=False,
            name=name + "_delta",
        )
        delta = train_NPML_model(
            context=context,
            model=delta_model,
            sample_dataset=sample_dataset,
            reference_dataset=reference_dataset,
        )
    else:
        delta_model = None
        delta = 0

    # Calculate t test statistic
    t = tau - delta

    return tau_model, t
