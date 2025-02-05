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

from neural_networks.NPLM.src.NPLM.NNutils import *
from neural_networks.NPLM.src.NPLM.PLOTutils import *
from neural_networks.NPLM.src.NPLM.ANALYSISutils import *
from fractions import Fraction
from read_h5_IS import *
from new_setting import *


parser = argparse.ArgumentParser()
parser.add_argument("-j", "--jobid", help="job id", dest="jobid")
parser.add_argument("-R", "--train-size", help="train size", dest="train_size")
parser.add_argument("-B", "--test-size", help="test size", dest="test_size")
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
parser.add_argument("-S","--spl", help="decides which samples to use (em or exp)", dest="sample_str")




args = parser.parse_args()

jobid = args.jobid
train_size = args.train_size
test_size = args.test_size
sig_events = args.sig_events
jsonfile = args.jsonfile
outdir = args.outdir
seed = args.seed

np.random.seed(seed)

## Import parameters from the given json file
with open(jsonfile, 'r') as js:
    config = json.load(js)

with open('/srv01/tgrp/inbarsav/LFV_git/LFV_nn/LFV_nn/initial_config.json', 'r') as js:
    initial_config = json.load(js)

# nuisance
correction = config['correction']
Scale      = config["shape_nuisances_reference"]
Norm       = config["norm_nuisances_reference"]
sigma_s    = config['shape_nuisances_sigma']
sigma_n    = config['norm_nuisances_sigma']

#gaussian signal parameters
Sig_loc = ''
Sig_scale = ''

# training time
epochs_tau   = config['epochs_tau']
patience_tau       = config['patience_tau']
epochs_delta = config['epochs_delta']
patience_delta     = config['patience_delta']

# architecture
BSMweight_clipping = config['BSMweight_clipping']
BSMarchitecture    = config['BSMarchitecture']
inputsize          = BSMarchitecture[0]
BSMdf              = compute_df(input_size=BSMarchitecture[0], hidden_layers=BSMarchitecture[1:-1])



sample_str = args.sample_str

Bkg_Aux_str = args.Bkg_Aux_str
Sig_Aux_str = args.Sig_Aux_str
Bkg_Data_str = args.Bkg_Data_str
Sig_Data_str = args.Sig_Data_str
sample_pdf = args.sample_pdf


CP = config['combined_portion']
sample = em if sample_str=='em' else exp if sample_str=='exp' else 'function'

resample_str = config['resample']
if resample_str=="True":
    label_method = config['label_method']
    N_method = config['N_method']
    resample_seed = int(seed)
    seed = int(config['original_seed'])
    replacement = config["replacement"]=="True"


print('before prepare')
if sample_str=='em':
    N_poiss = config['N_poiss']=="True"
    feature,target = prepare_training(em,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,train_size = train_size,test_size = test_size, sig_events = sig_events,seed = seed,N_poiss = N_poiss,combined_portion=CP)
if sample_str=='exp':
    Sig_loc = config['Sig_loc']
    Sig_scale = config['Sig_scale']
    N_poiss = config['N_poiss']=="True"
    resonant = config['resonant']=='True'
    feature,target = prepare_training(exp,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,N_Ref=round(219087*float(Fraction(train_size))*CP),N_Bkg=round(219087*float(Fraction(test_size))*CP),N_Sig = sig_events,seed = seed,Scale=Scale,Norm=Norm,Sig_loc=Sig_loc,Sig_scale=Sig_scale, N_poiss=N_poiss, resonant=resonant)
print('after prepare')

if resample_str=="True":
    N_A = feature[target[:, 0]==0].shape[0]
    feature,target = resample(feature = feature,target = target,Bkg_Data_str = Bkg_Data_str ,resample_seed = resample_seed,label_method=label_method, N_method = N_method, replacement = replacement)
    seed = resample_seed
batch_size  = feature.shape[0]
inputsize   = feature.shape[1]

np.random.seed(seed)
##NN model arguments
NU0_S     = np.random.normal(loc=Scale, scale=sigma_s, size=1)[0]
NU0_N     = np.random.normal(loc=Norm, scale=sigma_n, size=1)[0]
NUR_S     = np.array([0. ])
NUR_N     = 0
NU_S      = np.array([0. ])
NU_N      = 0
SIGMA_S   = np.array([sigma_s])
SIGMA_N   = sigma_n

input_shape = (None, inputsize)


# ##Taylor's expansion models for shape effects
# weights_file  = '/storage/agrp/yuvalzu/NPLM/NPLM_package/example_1D/LINEAR_Parametric_EXPO1D_batches_ref40000_bkg40000_sigmaS0.1_-1.0_-0.5_0.5_1.0_patience300_epochs30000_layers1_4_1_actrelu_model_weights9300.h5'

# sigma = weights_file.split('sigmaS', 1)[1]
# sigma = float(sigma.split('_', 1)[0])
# scale_list=weights_file.split('sigma', 1)[1]
# scale_list=scale_list.split('_patience', 1)[0]
# scale_list=np.array([float(s) for s in scale_list.split('_')[1:]])*sigma
# print(scale_list)
# shape_std = np.std(scale_list)
# activation= weights_file.split('act', 1)[1]
# activation=activation.split('_', 1)[0]
# wclip=None
# if 'wclip' in weights_file:
#     wclip= weights_file.split('wclip', 1)[1]
#     wclip = float(wclip.split('/', 1)[0])

# layers=weights_file.split('layers', 1)[1]
# layers= layers.split('_act', 1)[0]
# architecture = [int(l) for l in layers.split('_')]

# poly_degree = 0
# if 'LINEAR' in weights_file:
#     poly_degree = 1
# elif 'QUADRATIC' in weights_file:
#     poly_degree = 2
# elif 'CUBIC' in weights_file:
#     poly_degree = 3
# else:
#     print('Unrecognized number of degree for polynomial parametric net in file: \n%s'%(weights_file))
#     poly_degree = None

# scale_parNN = {
#     'poly_degree'   : poly_degree,
#     'architectures' : [architecture for i in range(poly_degree)],
#     'wclips'        : [wclip for i in range(poly_degree)],
#     'activation'    : activation,
#     'shape_std'     : shape_std,
#     'weights_file'  : weights_file
#     }

# parNN_list = {
#     'scale': scale_parNN,
#     }

#used for plotting the samples, not needed for this code.
REF    = feature[target[:, 0]==0]
DATA   = feature[target[:, 0]==1]
weight = target[:, 1]
weight_REF       = weight[target[:, 0]==0]
weight_DATA      = weight[target[:, 0]==1]


OUTPUT_PATH    = f"{outdir}/"#'/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/'
OUTPUT_FILE_ID = sample_str+''.join(list((f"{config[key]}{key}" if config[key]!=initial_config[key] else '' for key in config.keys())))+f'{sig_events}signals_{train_size}Ref_{test_size}Bkg_job{jobid}_seed'.replace('/',':')+str(seed)
if sample_pdf:
    OUTPUT_FILE_ID = "pdf_"+OUTPUT_FILE_ID
    
t = args.t
## Get Tau term model
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
hist_t_model = t_model.fit(feature, target, batch_size=batch_size, epochs=epochs_delta, verbose=0)
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
log_t = OUTPUT_PATH+OUTPUT_FILE_ID+f'_{t}.txt'
out   = open(log_t,'w')
out.write("%f\n" %(t_model_OBS))
out.close()
print("\nsaved t")

# save the training history                                       
log_history = OUTPUT_PATH+OUTPUT_FILE_ID+f'_{t}_history.h5'
f           = h5py.File(log_history,"w")
epoch       = np.array(range(epochs_delta))
patience_t = patience_tau if t=='TAU' else patience_delta if t=='delta' else 1000
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