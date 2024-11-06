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
from read_h5_IS import *
from new_setting_table import *


## Parse arguments (from the command line, used through the submit script.)
parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jobid", help="job id", dest="jobid")
parser.add_argument("-A", "--A-size", help="size of sample A", dest="A_size")
parser.add_argument("-B", "--B-size", help="size of sample B", dest="B_size")
parser.add_argument("-s", "--sig", help="number of signal events", type=int, dest="sig_events")
parser.add_argument("-c", "--jsonfile", help="json file", dest="jsonfile")
parser.add_argument("-o", "--outdir", help="output directory", dest="outdir")
parser.add_argument("--seed", help="seed", type=int, dest="seed")
parser.add_argument("-t", "--loss", help="string before history/weights.h5 and .txt names (TAU or delta)", dest="t")
parser.add_argument("--BAstr", help="string of data sets in the auxillary background (e.g. 'Ref' or 'Sig+Bkg+Ref')", default = '' ,dest="Bkg_Aux_str")
parser.add_argument("--SAstr", help="string of data sets in the auxillary sig (e.g. '' or 'Sig+Bkg')", default = '', dest="Sig_Aux_str")
parser.add_argument("--BDstr", help="string of data sets in the data background (e.g. 'Bkg' or 'Sig+Bkg')", default = '', dest="Bkg_Data_str")
parser.add_argument("--SDstr", help="string of data sets in the data sig (e.g. 'Sig' or 'Sig+Bkg')", default = '', dest="Sig_Data_str")
parser.add_argument("--smlpdf", help="if 1: generate data from pdf", type=int, dest="sample_pdf")


args = parser.parse_args()
jobid = args.jobid
A_size = args.A_size
B_size = args.B_size
sig_events = args.sig_events
jsonfile = args.jsonfile
outdir = args.outdir
seed = args.seed
Bkg_Aux_str = args.Bkg_Aux_str
Sig_Aux_str = args.Sig_Aux_str
Bkg_Data_str = args.Bkg_Data_str
Sig_Data_str = args.Sig_Data_str
sample_pdf = args.sample_pdf

np.random.seed(seed)

## Import parameters from the given json file
with open(jsonfile, 'r') as js:
    config = json.load(js)

initial_jsonfile = '/srv01/agrp/yuvalzu/scripts/NPLM_package/new_initial_config.json' if 'yuvalzu' in outdir else '/srv01/tgrp/inbarsav/LFV_git/LFV_nn/LFV_nn/new_initial_config.json'
## Initial json is used to compare the current json parameters to the standard ones, will mostly affect the output file name. 
with open(initial_jsonfile, 'r') as js:
    initial_config = json.load(js)

## Load json parameters
channel = config['ch']    # em, exp...
variables = config['vars'].split(':')    # string like 'Mcoll:Lep0Pt'..., becomes a list of strings.
signal_types = config['sig_types'].split(':')    # string like 'ggH_taue:vbfH_taumu'..., becomes a list of strings.
normalization = config['norm']    # number for dividing the data by, or str of the normalization method.

# nuisance
correction = ""
Scale      = [0]
Norm       = [0]
sigma_s    = [0.15]
sigma_n    = 0.15

# gaussian signal parameters
Sig_loc = ''
Sig_scale = ''

# training time
epochs   = config['epochs']
patience = config['patience']

# architecture
BSMweight_clipping = config['WC']
BSMarchitecture    = [int(i) for i in config['architecture'].split(':')]
# if len(variables)!=BSMarchitecture[0]:
#     raise ValueError(f"Number of variables ({len(variables)}) does not match the input layer size ({BSMarchitecture[0]})")
if BSMarchitecture[-1]!=1:
    raise ValueError(f"Output layer size ({BSMarchitecture[-1]}) must be 1")
inputsize          = BSMarchitecture[0]
BSMdf              = compute_df(input_size=BSMarchitecture[0], hidden_layers=BSMarchitecture[1:-1], output_size=BSMarchitecture[-1])

N_poiss = config['poiss']=="True"
CP = config['CP']    # combined portion, between 0 and 1. total number of events (as a fraction) in the over-all sample (before splitting into A and B).
if 'binned' in config:
    binned = config['binned']=="True"
if 'resolution' in config:
    resolution = config['resolution']

resample_str = config['resample']
if resample_str=="True":
    label_method = config['label_method']
    N_method = config['N_method']
    resample_seed = int(seed)
    seed = int(config['original_seed'])
    replacement = config["replacement"]=="True"

print('before prepare')
if channel=='exp':
    Sig_loc = config['Sig_loc']
    Sig_scale = config['Sig_scale']
    resonant = config['resonant']=='True'
    feature,target = prepare_training(exp,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,
                                      N_A=round(219087*float(Fraction(A_size))*CP),N_B=round(219087*float(Fraction(B_size))*CP),N_Sig = sig_events,
                                      seed = seed,Sig_loc=Sig_loc,Sig_scale=Sig_scale, N_poiss=N_poiss, resonant=resonant)
elif channel=='gauss':
    Sig_loc = config['Sig_loc']
    Sig_scale = config['Sig_scale']
    NR = config['NR']
    ND = config['ND']
    feature,target = prepare_training(gauss,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,
                                      N_A=round(219087*float(Fraction(A_size))*CP),N_B=round(219087*float(Fraction(B_size))*CP),N_Sig = sig_events,
                                      seed = seed,Sig_loc=Sig_loc,Sig_scale=Sig_scale, N_poiss=N_poiss, NR=NR, ND=ND)
else:
    feature,target = prepare_training(physics,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,
                                      A_size=A_size,B_size=B_size,sig_events=sig_events,
                                      seed=seed,channel=channel,signal_types=signal_types,phys_variables=variables,N_poiss=N_poiss,combined_portion=CP,
                                      binned=binned,resolution=resolution,normalization=normalization)
if resample_str=="True":
    N_A = feature[target[:, 0]==0].shape[0]
    feature,target = resample(feature = feature,target = target,Bkg_Data_str = Bkg_Data_str ,resample_seed = resample_seed,label_method=label_method, N_method = N_method, replacement = replacement)
    seed = resample_seed
    
print('after prepare')
batch_size  = feature.shape[0]
inputsize   = feature.shape[1]

## NN model arguments
NU0_S     = np.random.normal(loc=Scale, scale=sigma_s, size=1)[0]
NU0_N     = np.random.normal(loc=Norm, scale=sigma_n, size=1)[0]
NUR_S     = np.array([0. ])
NUR_N     = 0
NU_S      = np.array([0. ])
NU_N      = 0
SIGMA_S   = np.array([sigma_s])
SIGMA_N   = sigma_n

input_shape = (None, inputsize)

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## used for plotting the events distribution, not needed for this code (training).
REF    = feature[target[:, 0]==0]
DATA   = feature[target[:, 0]==1]
weight = target[:, 1]
weight_REF       = weight[target[:, 0]==0]
weight_DATA      = weight[target[:, 0]==1]
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


OUTPUT_PATH    = f"{outdir}/"
OUTPUT_FILE_ID = ''.join(list((f"{key}-{config[key]}_" if config[key]!=initial_config[key] else '' for key in config.keys())))+f'{sig_events}signals_{A_size}Ref_{B_size}Bkg_job{jobid}_seed'.replace('/',':')+str(seed)    # creats output file name from the json parameters
    
t = args.t
## Create the NN's model
t_model = imperfect_model(input_shape=input_shape,
                      NU_S=NUR_S, NUR_S=NUR_S, NU0_S=NU0_S, SIGMA_S=SIGMA_S, 
                      NU_N=NUR_N, NUR_N=NUR_N, NU0_N=NU0_N, SIGMA_N=SIGMA_N,
                      correction=correction, 
                      #shape_dictionary_list=[parNN_list['scale']],
                      BSMarchitecture=BSMarchitecture, BSMweight_clipping=BSMweight_clipping, train_f=True, train_nu=False)
print(t_model.summary())

t_model.compile(loss=imperfect_loss,  optimizer='adam')

## Train
print("\nStarting training")
t0=time.time()
hist_t_model = t_model.fit(feature, target, batch_size=batch_size, epochs=epochs, verbose=0)
t1=time.time()
print(f'Training time (seconds): {t1-t0} \n')

## Save

# metrics                      
loss_t_model  = np.array(hist_t_model.history['loss'])

# test statistic                                         
final_loss = loss_t_model[-1]
t_model_OBS    = -2*final_loss    # observed test score: t=-2L
print('t_model_OBS: %f'%(t_model_OBS))

# save t                                                                                                               
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+f'_{t}.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_model_OBS))
out.close()
print("\nsaved t")

# save the training history                                       
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+f'_{t}_history.h5'
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

# save the model    
log_weights = OUTPUT_PATH+OUTPUT_FILE_ID+f'_{t}_weights.h5'
t_model.save_weights(log_weights)
print(log_weights)
print("\nsaved weights")