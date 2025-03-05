import os, time, h5py
from neural_networks.NPLM_adapters import build_feature_for_model_train, build_shape_dictionary_list, build_target_for_model_loss
import numpy as np
import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
from tensorflow import linalg as la

from neural_networks.NPLM.src.NPLM.NNutils import *
from neural_networks.NPLM.src.NPLM.PLOTutils import *
from neural_networks.NPLM.src.NPLM.ANALYSISutils import *
from data_tools.data_utils import DataSet, prepare_training, resample
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import SINGLE_TRAINING_RESULT_FILE_NAME, TRAINING_HISTORY_FILE_NAME, TRIANING_OUTCOMES_DIR_NAME, WEIGHTS_OUTPUT_FILE_NAME
from train.train_config import TrainConfig

@context_controlled_execution
def main(context: ExecutionContext) -> None:
    # Naming shortcut
    config = context.config

    # type casting safety for the config type
    if not isinstance(config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    # Prepare sample
    exp_dataset, aux_dataset = prepare_training(config)
    feature_dataset = build_feature_for_model_train(exp_dataset, aux_dataset)

    if config.train__resample_is_resample:  # todo: this was never checked
        raise NotImplementedError("Resampling is not implemented")
        feature_dataset, target_structure = resample(
            feature = feature_dataset,
            target = target_structure,
            background_data_str = config.train__data_background ,
            label_method = config.train__resample_label_method,
            method_type = config.train__resample_method_type,
            replacement = config.train__resample_is_replacement,
        )

    t_model_OBS, t_model_history, t_model = \
        train_for_t_using_nn(
            config=config,
            feature_dataset=feature_dataset,
            exp_dataset=exp_dataset,
            aux_dataset=aux_dataset,
        )

    save_training_outcomes(context, t_model_OBS, t_model_history, t_model)


def train_for_t_using_nn(
        config: TrainConfig,
        feature_dataset: DataSet,
        # target_structure: np.ndarray):
        exp_dataset: DataSet,
        aux_dataset: DataSet,
    ):
    ## NN model arguments
    
    # Treating nuisance parameters
    ## normalization of the nuisance parameters, $\nu_n$ in the text
    SIGMA_N   = config.train__nuisances_norm_std
    NU_N      = config.train__nuisances_norm_mean * SIGMA_N
    NUR_N     = config.train__nuisances_norm_reference * SIGMA_N
    NU0_N     = np.random.normal(loc=NU_N, scale=SIGMA_N, size=1)[0]

    ## shape of the nuisance parameters, $\nu_s$ in the text
    SIGMA_S   = np.array([config.train__nuisances_shape_std])
    NU_S      = np.array([config.train__nuisances_shape_mean * SIGMA_S])
    NUR_S     = np.array([config.train__nuisances_shape_reference * SIGMA_S])
    NU0_S     = np.random.normal(loc=NU_S[0], scale=SIGMA_S[0], size=1)[0]

    # Done preparing sample
    batch_size  = feature_dataset.n_samples
    input_size  = feature_dataset._data.shape[1]  # Number of input variables

    # Get Tau term model
    tau_model = imperfect_model(
        input_shape=(None, input_size),
        NU_S=NU_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S,  # Lists of parameters
        NU_N=NU_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,  # integers
        correction = config.train__nuisance_correction,
        # shape_dictionary_list = build_shape_dictionary_list(),  # todo: what with this?
        BSMarchitecture = config.train__nn_architecture,
        BSMweight_clipping = config.train__nn_weight_clipping,
        train_f = True,  # = Should create model.BSMfinderNet
        train_nu = False,
    )
    logging.info(tau_model.summary())

    tau_model.compile(loss=imperfect_loss,  optimizer='adam')

    # Train
    target_structure = build_target_for_model_loss(exp_dataset, aux_dataset)

    logging.debug("Starting training")
    t0=time.time()
    t_model_history = tau_model.fit(  # todo: NPLM uses their own train_model instead of model.fit. Why? [[toy_batch.py]]
        np.array(feature_dataset._data, dtype=np.float32),
        np.array(target_structure, dtype=np.float32),
        batch_size=batch_size,
        epochs=config.train__epochs,
        verbose=0,
    )
    logging.debug(f'Training time (seconds): {time.time() - t0}')

    loss_t_model  = np.array(t_model_history.history['loss'])                
    final_loss = loss_t_model[-1]
    t_model_OBS    = -2 * final_loss
    logging.info('t_model_OBS (test statistic): %f'%(t_model_OBS))
    return t_model_OBS, t_model_history, tau_model


def save_training_outcomes(
        context: ExecutionContext,
        t_model_OBS: float,
        t_model_history: tf.keras.callbacks.History,
        t_model: Model
    ) -> None:
    ## Training log
    out_dir = context.unique_out_dir / TRIANING_OUTCOMES_DIR_NAME
    os.makedirs(out_dir, exist_ok=False)
    with open(out_dir / SINGLE_TRAINING_RESULT_FILE_NAME, 'w') as training_log:
        training_log.write("%f\n" %(t_model_OBS))

    ## Training history
    with h5py.File(out_dir / TRAINING_HISTORY_FILE_NAME,"w") as history_file:
        epoch       = np.array(range(context.config.train__epochs))
        patience_t = context.config.train__patience
        keepEpoch   = epoch % patience_t == 0
        history_file.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
        for key in list(t_model_history.history.keys()):
            monitored = np.array(t_model_history.history[key])
            logging.debug('%s: %f'%(key, monitored[-1]))
            history_file.create_dataset(key, data=monitored[keepEpoch], compression='gzip')
        logging.info("saved history")

    # save the model weights
    log_weights = out_dir / WEIGHTS_OUTPUT_FILE_NAME
    t_model.save_weights(log_weights)


if __name__ == "__main__":
    main()
