from logging import config
import signal
import sys, os, time, datetime, h5py, json, argparse
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.constraints import Constraint
from tensorflow.keras import metrics, losses, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow import Variable
from tensorflow import linalg as la

from NPLM.NNutils import *
from NPLM.PLOTutils import *
from NPLM.ANALYSISutils import *
from fractions import Fraction
from data_tools.profile_likelihood import *
from data_tools.histogram_generation import *
from frame.command_line.handle_args import context_controlled_execution
from frame.config_handle import ExecutionContext
from frame.file_structure import TRAINING_LOG_FILE_NAME, TRAINING_HISTORY_FILE_NAME, OUTPUT_FILE_NAME, TRIANING_OUTCOMES_DIR_NAME, WEIGHTS_OUTPUT_FILE_NAME
from train.train_config import TrainConfig
from neural_networks.NNutils import ImperfectModel, imperfect_loss
from configs.config_utils import parNN_list

@context_controlled_execution
def main(context: ExecutionContext) -> None:
    # Naming shortcut
    config = context.config

    # type casting safety for the config type
    if not isinstance(config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    # training time settings
    if config.train__epochs_type == 'TAU' or config.train__epochs_type == 'delta':
        epochs = config.train__epochs
        patience = config.train__patience
    else:
        epochs = max(config.train__epochs, config.train__epochs)
        patience = min(config.train__patience, config.train__patience)

    # background = np.load(config.train__data_dir / config.train__backgournd_distribution_path)
    # signal = np.load(config.train__data_dir / config.train__signal_distribution_path)

    # Prepare sample
    feature, target = prepare_training(
        config.analytic_background_function,
        config.train__data_background_aux,
        config.train__data_signal_aux,
        config.train__data_background,
        config.train__data_signal,
        NR = "False",  # config.NR
        ND = "False",  # config.ND
        n_signal = config.train__signal_number_of_events,
        **config.analytic_background_function_calling_parameters,
    )

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

    ## Get Tau term model
    t_model = ImperfectModel(
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
    print(t_model.summary())

    t_model.compile(loss=imperfect_loss,  optimizer='adam')

    ## Train
    print("\nStarting training")
    t0=time.time()
    t_mdoel_history = t_model.fit(feature, target, batch_size=batch_size, epochs=epochs, verbose=0)
    t1=time.time()
    print('Training time (seconds):')
    print(t1-t0,"\n")

    ## Save

    # metrics                      
    loss_t_model  = np.array(t_mdoel_history.history['loss'])

    # test statistic                                         
    final_loss = loss_t_model[-1]
    t_model_OBS    = -2*final_loss
    print('t_model_OBS: %f'%(t_model_OBS))

    out_dir = context.unique_out_dir / TRIANING_OUTCOMES_DIR_NAME
    os.makedirs(out_dir, exist_ok=False)

    # save t
    with open(out_dir / TRAINING_LOG_FILE_NAME, 'w') as log:
        log.write("%f\n" %(t_model_OBS))

    # save the training history                                       
    log_history = TRAINING_HISTORY_FILE_NAME
    f           = h5py.File(out_dir / log_history,"w")
    epoch       = np.array(range(epochs))
    patience_t = patience
    keepEpoch   = epoch % patience_t == 0
    f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
    for key in list(t_mdoel_history.history.keys()):
        monitored = np.array(t_mdoel_history.history[key])
        print('%s: %f'%(key, monitored[-1]))
        f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
    f.close()
    print("\nsaved history")

    # save the model weights
    log_weights = out_dir / WEIGHTS_OUTPUT_FILE_NAME
    t_model.save_weights(log_weights)

if __name__ == "__main__":
    main()
