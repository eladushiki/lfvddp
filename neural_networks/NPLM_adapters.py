from logging import debug, info
from time import time
from typing import Any, Dict, Union
from data_tools.data_utils import DataSet
from data_tools.dataset_config import DatasetConfig
from data_tools.profile_likelihood import calc_t_test_statistic
from frame.context.execution_context import ExecutionContext
from frame.file_structure import TRAINING_HISTORY_FILE_EXTENSION, WEIGHTS_OUTPUT_FILE_NAME
from neural_networks.NPLM.src.NPLM.ANALYSISutils import h5py, np, os
from neural_networks.NPLM.src.NPLM.NNutils import h5py, imperfect_loss, imperfect_model, logging, np, train_model
from neural_networks.NPLM.src.NPLM.PLOTutils import h5py, np, os
import numpy as np
from tensorflow.keras import optimizers # type: ignore

from neural_networks.weights.taylor_expansion_net.parameters import parNN_list
from train.train_config import TrainConfig


def build_feature_for_model_train(exp_dataset, aux_dataset):
    return exp_dataset + aux_dataset


def build_target_for_model_loss(sample_dataset: DataSet, reference_dataset: DataSet):
    ## target structure
    ones_like_sample = np.ones_like(sample_dataset, shape=(sample_dataset.n_samples, 1))  # 1 for dim 1 because the NN's output is 1D.
    zeros_like_reference = np.zeros_like(reference_dataset, shape=(reference_dataset.n_samples, 1))
    reference_weights = np.ones_like(reference_dataset, shape=(reference_dataset.n_samples, 1)) \
        * sample_dataset.n_samples * 1. / reference_dataset.n_samples

    is_sample_mask = np.concatenate((ones_like_sample, zeros_like_reference), axis=0)
    is_sample_with_reference_weights = np.concatenate((ones_like_sample, reference_weights), axis=0)
    loss_mask = np.concatenate((is_sample_mask, is_sample_with_reference_weights), axis=1)
    
    return loss_mask


def build_shape_dictionary_list():
    # todo: this should be drawn from the config
    return [parNN_list['scale']]  # todo: this should be of the length of deltas? Look @ imperfect_model implementation


def get_tau_predicting_model(config: Union[DatasetConfig, TrainConfig], name: str = "tau_model") -> imperfect_model:
    """
    Generate an NPLM imperfect model according to our configuration
    """
    # Solely for type hinting to take place
    if not isinstance(config, DatasetConfig):
        raise TypeError(f"Expected DatasetConfig, got {config.__class__.__name__}")
    if not isinstance(config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    
    ## Treating nuisance parameters
    # normalization of the nuisance parameters, $\nu_n$ in the text
    SIGMA_N   = config.dataset__nuisances_norm_sigma
    NU_N      = config.dataset__nuisances_norm_mean_sigmas * SIGMA_N
    NUR_N     = config.dataset__nuisances_norm_reference_sigmas * SIGMA_N
    NU0_N     = np.random.normal(loc=NU_N, scale=SIGMA_N, size=1)[0]

    # shape of the nuisance parameters, $\nu_s$ in the text
    SIGMA_S   = np.array([config.dataset__nuisances_shape_sigma])
    NU_S      = np.array([config.dataset__nuisances_shape_mean_sigmas * SIGMA_S])
    NUR_S     = np.array([config.dataset__nuisances_shape_reference_sigmas * SIGMA_S])
    NU0_S     = np.random.normal(loc=NU_S[0], scale=SIGMA_S[0], size=1)[0]

    # Get Tau term model
    tau_model = imperfect_model(
        name=name,
        input_shape=(None, config.train__nn_input_dimension),
        NU_S=NU_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S,  # Lists of parameters
        NU_N=NU_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,  # integers
        correction = config.train__nuisance_correction,
        shape_dictionary_list = build_shape_dictionary_list(),  # This is used in "SHAPE" correction case
        BSMarchitecture = config.train__nn_architecture,
        BSMweight_clipping = config.train__nn_weight_clipping,
        train_f = True,  # = Should create model.BSMfinderNet = is training also for Tau (else, just Delta). We generally want to train for both.
        train_nu = config.train__data_is_train_for_nuisances,
    )
    info(tau_model.summary())

    return tau_model


def train_model_for_tau(
        context: ExecutionContext,
        tau_model: imperfect_model,
        sample_dataset: DataSet,
        reference_dataset: DataSet
    ) -> float:
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")
    
    ## Done preparing sample
    feature_dataset = build_feature_for_model_train(sample_dataset, reference_dataset)
    target_structure = build_target_for_model_loss(sample_dataset, reference_dataset)

    # Train
    debug("Starting training")
    t0 = time()
    tau_model_history = train_model(
        model=tau_model,
        feature=np.array(feature_dataset._data, dtype=np.float32),
        target=np.array(target_structure, dtype=np.float32),
        loss=imperfect_loss,  # This is (11) in "Learning New Physics from a Machine", D'Angolo et al.
        optimizer=optimizers.legacy.Adam(),
        total_epochs=config.train__epochs,
        patience=config.train__number_of_epochs_for_checkpoint,
        clipping=config.train__nn_weight_clipping > 0,
        verbose=False,
    )
    tau_history = np.array(tau_model_history['loss'])                
    debug(f'Training time (seconds): {time() - t0}')

    final_t = calc_t_test_statistic(tau_history[-1])
    logging.info(f'Observed t test statistic: {final_t}')
    
    save_NPLM_training_outcomes(
        context,
        tau_model_history=tau_model_history,
        tau_model=tau_model,
    )

    return final_t


def save_NPLM_training_outcomes(
        context: ExecutionContext,
        tau_model_history: Dict[str, Any],
        tau_model: imperfect_model,
    ) -> None:
    if not isinstance(config := context.config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    ## Training log
    os.makedirs(context.training_outcomes_dir, exist_ok=True)

    ## Training history
    history_path = context.training_outcomes_dir / f"{tau_model.name}.{TRAINING_HISTORY_FILE_EXTENSION}"
    with h5py.File(history_path,"w") as history_file:
        for key in list(tau_model_history.keys()):
            monitored = np.array(tau_model_history[key])
            logging.debug(f'{key}: {monitored[-1]}')
            history_file.create_dataset(key, data=monitored, compression='gzip')
    context.document_created_product(history_path)

    # save the model weights
    context.save_and_document_model_weights(tau_model, context.training_outcomes_dir / f"{tau_model.name}_{WEIGHTS_OUTPUT_FILE_NAME}")
