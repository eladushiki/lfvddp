from logging import config
import signal
import sys, os, time, datetime, h5py, json, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.stats import norm, expon, chi2, uniform, chisquare
import scipy.stats as sps
from sklearn.model_selection import train_test_split

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
from frame.command_line_args import context_controlled_execution
from frame.config_handle import ExecutionContext
from frame.file_structure import LOG_FILE_NAME, LOG_HISTORY_FILE_NAME, OUTPUT_FILE_NAME, WEIGHTS_FILE_NAME
from train.train_config import TrainConfig

@context_controlled_execution
def main(context: ExecutionContext) -> None:
    config = context.config

    # type casting safety for the config type
    if not isinstance(config, TrainConfig):
        raise TypeError(f"Expected TrainConfig, got {config.__class__.__name__}")

    # training time settings
    if config.train__epochs_type == 'TAU':
        epochs = config.train__epochs
        patience = config.train__patience
    elif config.train__epochs_type == 'delta':
        epochs = config.train__epochs
        patience = config.train__patience
    else:
        epochs = max(config.train__epochs, config.train__epochs)
        patience = min(config.train__patience, config.train__patience)

    background = np.load(config.train__data_dir / config.train__backgournd_distribution_path)
    signal = np.load(config.train__data_dir / config.train__signal_distribution_path)

    if config.train__histogram_analytic_pdf == 'em' or \
        config.train__histogram_analytic_pdf == 'em_Mcoll': # TODO: there should be a difference in scale between both
        analytic_background_function = em
        kwargs = {
            "background_distribution": background,
            "signal_distribution": signal,
            "binned": config.train__histogram_is_binned,
            "resolution": config.train__histogram_resolution,
        }
    elif config.train__histogram_analytic_pdf == 'exp':
        analytic_background_function = exp
        kwargs = {
            "N_Ref": round(219087*float(Fraction(config.train__batch_size)) * config.train__combined_portion),
            "N_Bkg": round(219087*float(Fraction(config.train__test_batch_size)) * config.train__combined_portion),
            "N_sig": config.train__signal_number_of_events,
            "Scale": config.train__nuisance_scale,
            "Sig_loc": config.train__signal_location,
            "Sig_scale": config.train__signal_scale,
            "resonant": config.train__signal_resonant,
        }
    else:
        raise ValueError(f"Invalid sample_string: {config.train__histogram_analytic_pdf}")

    # Prepare sample
    feature, target = prepare_training(
        analytic_background_function,
        config.train__data_background_aux,
        config.train__data_signal_aux,
        config.train__data_background,
        config.train__data_signal,
        train_size = config.train__batch_size,
        test_size = config.train__test_batch_size, 
        sig_events = config.train__signal_number_of_events,
        seed = context.random_seed,
        N_poiss = config.train__N_poiss,
        combined_portion=config.train__combined_portion,
        **kwargs,
    )

    # Done preparing sample
    batch_size  = feature.shape[0]
    inputsize   = feature.shape[1]

    # NN model arguments
    NU0_S     = np.random.normal(loc=config.train__nuisance_scale, scale=config.train__nuisance_sigma_s, size=1)[0]
    NU0_N     = np.random.normal(loc=config.train__nuisance_norm, scale=config.train__nuisance_sigma_n, size=1)[0]
    NUR_S     = np.array([0. ])
    NUR_N     = 0
    SIGMA_S   = np.array([config.train__nuisance_sigma_s])
    SIGMA_N   = config.train__nuisance_sigma_n

    input_shape = (None, inputsize)

    OUTPUT_PATH    = config.out_dir
    OUTPUT_FILE_ID = OUTPUT_FILE_NAME
    if config.train__histogram_is_use_analytic:
        OUTPUT_FILE_ID = "pdf_"+OUTPUT_FILE_ID
        
    ## Get Tau term model
    t_model = imperfect_model(input_shape=input_shape,
                        NU_S=NUR_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S, 
                        NU_N=NUR_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
                        correction=config.train__nuisance_correction, 
                        #shape_dictionary_list=[parNN_list['scale']],
                        BSMarchitecture=config.train__nn_architecture, BSMweight_clipping=config.train__nn_weight_clipping, train_f=True, train_nu=False)
    print(t_model.summary())

    t_model.compile(loss=imperfect_loss,  optimizer='adam')

    ## Train
    print("\nStarting training")
    t0=time.time()
    hist_t_model = t_model.fit(feature, target, batch_size=batch_size, epochs=epochs, verbose=0)
    t1=time.time()
    print('Training time (seconds):')
    print(t1-t0,"\n")

    ## Save

    # metrics                      
    loss_t_model  = np.array(hist_t_model.history['loss'])

    # test statistic                                         
    final_loss = loss_t_model[-1]
    t_model_OBS    = -2*final_loss
    print('t_model_OBS: %f'%(t_model_OBS))

    # save t
    with open(LOG_FILE_NAME, 'w') as log:
        log.write("%f\n" %(t_model_OBS))

    # save the training history                                       
    log_history = LOG_HISTORY_FILE_NAME
    f           = h5py.File(log_history,"w")
    epoch       = np.array(range(epochs))
    patience_t = patience
    keepEpoch   = epoch % patience_t == 0
    f.create_dataset('epoch', data=epoch[keepEpoch], compression='gzip')
    for key in list(hist_t_model.history.keys()):
        monitored = np.array(hist_t_model.history[key])
        print('%s: %f'%(key, monitored[-1]))
        f.create_dataset(key, data=monitored[keepEpoch],   compression='gzip')
    f.close()
    print("\nsaved history")

    # save the model weights
    log_weights = config.out_dir / WEIGHTS_FILE_NAME
    t_model.save_weights(log_weights)

if __name__ == "__main__":
    main()
