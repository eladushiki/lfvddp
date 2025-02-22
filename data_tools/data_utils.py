from enum import Enum
from typing import List

from regex import D
import numpy as np
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from fractions import Fraction
from random import choices

from train.train_config import TrainConfig


class DataGeneration:

    class DataSet(Enum):
        REF = "Ref"
        BKG = "Bkg"
        SIG = "Sig"

    def __init__(self, config: TrainConfig):
        self._config = config

        self._reference_dataset, self._background_dataset, self._signal_dataset = \
            self._config.train__analytic_background_function(config)
        
    def _generate_dataset(self, components: List[DataSet]):
        included_datasets = []

        if DataGeneration.DataSet.REF in components:
            included_datasets.append(self._reference_dataset)
        if DataGeneration.DataSet.BKG in components:
            included_datasets.append(self._background_dataset)
        if DataGeneration.DataSet.SIG in components:
            included_datasets.append(self._signal_dataset)

        return np.concatenate(included_datasets, axis=0)

    def generate_dataset(self, data_sets: List[str]):
        return self._generate_dataset([DataGeneration.DataSet(ds) for ds in data_sets])


def prepare_training(config: TrainConfig):
    '''
    Creates sets of featureData and featureRef according to featureData_str and featureRef_str respectively.
    
    Parameters
    ----------
    config :
        Includes the parameters necessary for training with the specific function.
    
    returns feature_dataset and target_structure
    '''
    # Generate datasets by config instructions
    dg = DataGeneration(config)
    aux_background_dataset = dg.generate_dataset(config.train__data_aux_background_composition)
    aux_signal_dataset = dg.generate_dataset(config.train__data_aux_signal_composition)
    exp_background_dataset = dg.generate_dataset(config.train__data_experimental_background_composition)
    exp_signal_dataset = dg.generate_dataset(config.train__data_experimental_signal_composition)
       
    feature_reference = np.concatenate((aux_background_dataset, aux_signal_dataset), axis=0)
    feature_data = np.concatenate((exp_background_dataset, exp_signal_dataset), axis=0)
    feature_dataset     = np.concatenate((feature_data, feature_reference), axis=0)

    ## target structure
    n_reference_samples = feature_reference.shape[0]
    n_background_samples = feature_data.shape[0]
    targetData       = np.ones_like(feature_data, shape=(feature_data.shape[0], 1))    # 1 for dim 1 because the NN's output is 1D.
    targetRef        = np.zeros_like(feature_reference, shape=(feature_reference.shape[0],1))
    weightsData      = np.ones_like(feature_data, shape=(feature_data.shape[0], 1))
    weightsRef       = np.ones_like(feature_reference, shape=(feature_reference.shape[0],1)) * n_background_samples * 1. / n_reference_samples
    target           = np.concatenate((targetData, targetRef), axis=0)
    weights          = np.concatenate((weightsData, weightsRef), axis=0)
    target_structure = np.concatenate((target, weights), axis=1)
    
    return feature_dataset, target_structure


def resample(
        feature: np.ndarray,
        target: np.ndarray,
        background_data_str: str = "Bkg",
        label_method: str = "permute",
        method_type: str = "fixed",
        replacement: bool = True
    ):
    '''
    Creates sets of featureData and featureRef according to featureData_str and featureRef_str respectively.
    
    Parameters
    ----------
    feature: ??
    target: ??
    background_data_str: = "Ref"|"Bkg"
    label_method: = "permute"|"binomial"|"bootstrap" 
        "permute": train_test_split according to current ratio (number of events in each sample remains unchanged)
        "binomial": determine label according to binomial distribution with p_A = N_A/(N_A+N_B) - only total number of events is fixed
        "bootstrap": resample uniformly (with/wo replacement) with sizes N_A and N_B 
    method_type: = "fixed"|"binomial"|"poiss"
    replacement : for bootstrap only - whether events can be repeated. True = repeat events, False - do not repeat events.

    Returns
    ----------
    feature, target
    '''
    combined_data = feature[target[:, 0]==0]
    N_combined = combined_data.shape[0]
    if "Ref" in background_data_str:
        N_B = feature[target[:, 0]==1].shape[0]
        N_A = N_combined-N_B
    else:
        N_A = feature[target[:, 0]==1].shape[0]
        N_B = N_combined-N_A
    
    if method_type=="poiss":
        N_A = np.random.poisson(lam=N_A, size=1)[0]
        N_B = np.random.poisson(lam=N_B, size=1)[0]
    
    if label_method=="permute":
        data_A, data_B = train_test_split(combined_data,train_size=N_A,test_size = N_B,random_state=resample_seed)
    elif label_method =="binomial":
        label = choices(["A","B"],weights = [N_A/(N_A+N_B),N_B/(N_A+N_B)],k = N_combined)
        data_A = combined_data[label=="A"]
        data_B = combined_data[label=="B"]
    elif label_method=="bootstrap":
        # Note: to have no replacement random.choice should actually be replaced by random.sample
        data_A = choices(combined_data, replace = replacement, k=N_A)
        data_B = choices(combined_data, replace = replacement, k=N_B)
    else:
        raise ValueError(f"Invalid label_method for resample: {label_method}")

    if "Ref" in background_data_str:
        featureData = data_B.copy()
        print("Ref")
    else:
        featureData = data_A.copy()
        print("Bkg")

    featureRef = np.concatenate((data_A,data_B),axis=0)
    N_ref = featureRef.shape[0]
    print(N_ref)
    feature     = np.concatenate((featureData, featureRef), axis=0)
    N_R        = N_ref
    N_D        = featureData.shape[0]#N_Bkg

    ## target
    targetData  = np.ones_like(featureData)
    targetRef   = np.zeros_like(featureRef)
    weightsData = np.ones_like(featureData)
    weightsRef  = np.ones_like(featureRef)*N_D*1./N_R
    target      = np.concatenate((targetData, targetRef), axis=0)
    weights     = np.concatenate((weightsData, weightsRef), axis=0)
    target      = np.concatenate((target, weights), axis=1)

    return feature,target


def em(
        background_distribution: np.ndarray,
        signal_distribution: np.ndarray,
        train_size,
        test_size,
        sig_events,
        seed,
        N_poiss=True,
        combined_portion=1,
        resolution=1,
        *args, **kwargs
    ):
    
    N_Bkg_Pois  = np.random.poisson(lam=float(Fraction(test_size))*background_distribution.shape[0]*combined_portion, size=1)[0] if N_poiss else float(Fraction(test_size))*background_distribution.shape[0]*combined_portion
    N_Ref_Pois  = np.random.poisson(lam=float(Fraction(train_size))*background_distribution.shape[0]*combined_portion, size=1)[0] if N_poiss else float(Fraction(train_size))*background_distribution.shape[0]*combined_portion
    N_Sig_Pois = np.random.poisson(lam=sig_events, size=1)[0] if N_poiss else sig_events
    
    Ref = np.random.choice(background_distribution.reshape(-1,),N_Ref_Pois,replace=True).reshape(-1,1)
    Bkg = np.random.choice(background_distribution.reshape(-1,),N_Bkg_Pois,replace=True).reshape(-1,1)
    total_Sig = np.concatenate(tuple(signal_distribution),axis=0).reshape(-1,)
    Sig = np.random.choice(total_Sig,N_Sig_Pois,replace=True).reshape(-1,1)

    # todo: I deleted here some 1e5 factors that divide all of these in the case of em and others for em_Mcoll
    # Need to go over it when I'll understand where the data comes from know the meaning of it
    return Ref,Bkg,Sig


def generate_pdf(x,seed,size=3e6):
    '''
    Not used (as of 9/23).
    Creates the pdf of the array x.
    Returns new array of length 'size' generated from the pdf of x. 
    '''
    np.random.seed(seed)
    full_dist = x
    print('creating histogram')
    full_dist_hist = np.histogram(x,len(full_dist))
    print('creating pdf')
    full_dist_pdf = sps.rv_histogram(full_dist_hist)
    print('creating new data')
    new_data = full_dist_pdf.rvs(size=size).reshape(-1,1)
    print('we have new data!')
    
    return new_data
