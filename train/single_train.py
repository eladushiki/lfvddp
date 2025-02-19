import os, time, h5py
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
from data_tools.data_utils import prepare_training, resample
from frame.command_line.handle_args import context_controlled_execution
from frame.context.execution_context import ExecutionContext
from frame.file_structure import SINGLE_TRAINING_RESULT_FILE_NAME, TRAINING_HISTORY_FILE_NAME, TRIANING_OUTCOMES_DIR_NAME, WEIGHTS_OUTPUT_FILE_NAME
from train.train_config import TrainConfig
from configs.config_utils import parNN_list

@context_controlled_execution
def main(context: ExecutionContext) -> None:
    # Naming shortcut
    config = context.config

    # type casting safety for the config type
    if not isinstance(config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    # Prepare sample
    feature, target = prepare_training(config)

    if config.train__resample_is_resample:
        feature,target = resample(
            feature = feature,
            target = target,
            background_data_str = config.train__data_background ,
            label_method = config.train__resample_label_method,
            method_type = config.train__resample_method_type,
            replacement = config.train__resample_is_replacement,
        )

    # Done preparing sample
    batch_size  = feature.shape[0]
    inputsize   = feature.shape[1]

    # NN model arguments
    NU0_S     = np.random.normal(loc=config.train__nuisance_scale, scale=config.train__nuisance_sigma_s, size=1)[0]
    NU0_N     = np.random.normal(loc=config.train__nuisance_norm, scale=config.train__nuisance_sigma_n, size=1)[0]
    NUR_S     = np.array([0. ])
    NUR_N     = 0
    NU_S      = np.array([0. ])
    NU_N      = 0
    SIGMA_S   = np.array([config.train__nuisance_sigma_s])
    SIGMA_N   = config.train__nuisance_sigma_n

    input_shape = (None, inputsize)

    # Get Tau term model
    t_model = imperfect_model(
        input_shape=input_shape,
        NU_S=NUR_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S, 
        NU_N=NUR_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
        correction = config.train__nuisance_correction,
        shape_dictionary_list = [parNN_list['scale']],
        BSMarchitecture = [int(layer_width) for layer_width in config.train__nn_architecture.split(":")],
        BSMweight_clipping=config.train__nn_weight_clipping,
        train_f=True,
        train_nu=False
    )
    logging.info(t_model.summary())

    t_model.compile(loss=imperfect_loss,  optimizer='adam')

    # Train
    logging.debug("Starting training")
    t0=time.time()
    t_mdoel_history = t_model.fit(feature, target, batch_size=batch_size, epochs=config.train__epochs, verbose=0)
    logging.debug(f'Training time (seconds): {time.time() - t0}')

    # metrics                      
    loss_t_model  = np.array(t_mdoel_history.history['loss'])                
    final_loss = loss_t_model[-1]
    t_model_OBS    = -2*final_loss
    logging.info('t_model_OBS (test statistic): %f'%(t_model_OBS))
    
    # Save
    ## Training log
    out_dir = context.unique_out_dir / TRIANING_OUTCOMES_DIR_NAME
    os.makedirs(out_dir, exist_ok=False)
    with open(out_dir / SINGLE_TRAINING_RESULT_FILE_NAME, 'w') as training_log:
        training_log.write("%f\n" %(t_model_OBS))

    ## Training history
    with h5py.File(out_dir / TRAINING_HISTORY_FILE_NAME,"w") as history_file:
        epoch       = np.array(range(config.train__epochs))
        patience_t = config.train__patience
        keepEpoch   = epoch % patience_t == 0
        history_file.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
        for key in list(t_mdoel_history.history.keys()):
            monitored = np.array(t_mdoel_history.history[key])
            logging.debug('%s: %f'%(key, monitored[-1]))
            history_file.create_dataset(key, data=monitored[keepEpoch], compression='gzip')
        logging.info("saved history")

    # save the model weights
    log_weights = out_dir / WEIGHTS_OUTPUT_FILE_NAME
    t_model.save_weights(log_weights)

if __name__ == "__main__":
    main()
