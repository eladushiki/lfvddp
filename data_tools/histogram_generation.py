from typing import List
import numpy as np
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from fractions import Fraction
from random import choices


def prepare_training(datasets,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,NR="False",ND="False",*args, **kwargs):
    '''
    Creates sets of featureData and featureRef according to featureData_str and featureRef_str respectively.
    
    Parameters
    ----------
    datasets : function
        represents the name of physical distribution- em or exp.
    
    Bkg_Aux_str : str
        what out of Ref, Bkg, Sig should be concatenated in featureData. Split with "+" (e.g. 'Ref' or 'Sig+Bkg+Ref').accepts empty strings. 

    Sig_Aux_str : str
        what out of Ref, Bkg, Sig should be concatenated in featureData. Split with "+" (e.g. '' or 'Sig+Bkg').accepts empty strings. 

    Bkg_Data_str : str
        what out of Ref, Bkg, Sig should be concatenated in featureData. Split with "+" (e.g. 'Bkg' or 'Sig+Bkg')".accepts empty strings. 

    Sig_Data_str : str
        what out of Ref, Bkg, Sig should be concatenated in featureData. Split with "+" (e.g. 'Sig' or 'Sig+Bkg')".accepts empty strings. 
    
    returns feature and target
    '''
    datasets_dict = {'Ref':np.array([]),'Bkg':np.array([]),'Sig':np.array([]),'':np.array([[]])}
    datasets_dict['Ref'],datasets_dict['Bkg'],datasets_dict['Sig'] = datasets(*args, **kwargs)
    Bkg_Aux = np.concatenate(tuple([datasets_dict[key] for key in Bkg_Aux_str.split('+',Bkg_Aux_str.count('+'))]),axis=0)
    Sig_Aux = np.concatenate(tuple([datasets_dict[key] for key in Sig_Aux_str.split('+',Sig_Aux_str.count('+'))]),axis=0) if Sig_Aux_str!='' else np.zeros((0,Bkg_Aux.shape[1]))
    Bkg_Data = np.concatenate(tuple([datasets_dict[key] for key in Bkg_Data_str.split('+',Bkg_Data_str.count('+'))]),axis=0)
    Sig_Data = np.concatenate(tuple([datasets_dict[key] for key in Sig_Data_str.split('+',Sig_Data_str.count('+'))]),axis=0) if Sig_Data_str!='' else np.zeros((0,Bkg_Data.shape[1]))
   
    N_Bkg = Bkg_Data.shape[0]
    featureData = np.concatenate((Bkg_Data,Sig_Data),axis=0)

    featureRef = np.concatenate((Bkg_Aux,Sig_Aux),axis=0)
    N_ref = featureRef.shape[0]

    feature     = np.concatenate((featureData, featureRef), axis=0)
    N_R        = N_ref if not isinstance(NR,int) else NR
    N_D        = featureData.shape[0] if not isinstance(ND,int) else ND#N_Bkg

    ## target
    targetData  = np.ones_like(featureData,shape=(featureData.shape[0],1))    # 1 for dim 1 because the NN's output is 1D.
    targetRef   = np.zeros_like(featureRef,shape=(featureRef.shape[0],1))
    weightsData = np.ones_like(featureData,shape=(featureData.shape[0],1))
    weightsRef  = np.ones_like(featureRef,shape=(featureRef.shape[0],1))*N_D*1./N_R
    target      = np.concatenate((targetData, targetRef), axis=0)
    weights     = np.concatenate((weightsData, weightsRef), axis=0)
    target      = np.concatenate((target, weights), axis=1)
    
    return feature,target


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


def exp(
        n_ref: int,
        n_bkg: int,
        n_signal: int,
        scale_factor: float = 0,
        normalization_factor: float = 0,
        signal_location: float = 6.4,
        signal_scale: float = 0.16,
        poisson_fluctuations: bool = True,
        is_resonant_signal_shape: bool = True
    ):
    '''
    Returns exponential samples A, B and Sig suitable for fitting.
    
    Parameters
    ----------
    n_ref: number of events in sample A.
    n_bkg: number of events in sample B.
    n_signal: number of events in the added signal.
    scale_factor: scale of the exponential distribution.
    normalization_factor: normalization of the exponential distribution.
    signal_location: mean of the gaussian distribution of the signal.
    signal_scale: standard deviation of the gaussian distribution of the signal.
    poisson_fluctuations: True - add poisson fluctuations to the number of events in each sample. False - do not add poisson fluctuations.
    is_resonant_signal_shape: True - add gaussian signal. False - add non-resonant signal.

    Returns
    -------
    A : numpy array
    B : numpy array
    Sig : numpy array
    '''
    N_Bkg_Pois  = np.random.poisson(lam=n_bkg*np.exp(normalization_factor), size=1)[0] if poisson_fluctuations else n_bkg
    N_Ref_Pois  = np.random.poisson(lam=n_ref*np.exp(normalization_factor), size=1)[0] if poisson_fluctuations else n_ref
    N_Sig_Pois = np.random.poisson(lam=n_signal*np.exp(normalization_factor), size=1)[0] if poisson_fluctuations else n_signal
    print(N_Bkg_Pois,N_Ref_Pois,N_Sig_Pois)

    Bkg = np.random.exponential(scale=np.exp(1*scale_factor), size=(N_Bkg_Pois, 1))
    Ref  = np.random.exponential(scale=1., size=(N_Ref_Pois, 1))
    if is_resonant_signal_shape:
        Sig = np.random.normal(loc=signal_location, scale=signal_scale, size=(N_Sig_Pois,1))*np.exp(scale_factor)
    else:
        def Sig_dist(x):
            dist = x**2*np.exp(-x)
            return dist/np.sum(dist)
        Sig = np.random.choice(np.linspace(0,100,100000),size=(N_Sig_Pois,1),replace=True,p=Sig_dist(np.linspace(0,100,100000)))*np.exp(scale_factor)   
    print(f'defs: exp, N_Ref={n_ref},N_Bkg={n_bkg},N_Sig={n_signal},Scale={scale_factor},Norm={normalization_factor},Sig_loc={signal_location},Sig_scale={signal_scale}, N_poiss = {poisson_fluctuations}, resonant = {is_resonant_signal_shape}')
    print('Ref',Ref.shape,'Bkg',Bkg.shape, 'Sig',Sig.shape)
    return Ref,Bkg,Sig


def gauss(
        n_ref: int,
        n_bkg: int,
        n_signal: int,
        normalization_factor: float,
        signal_location: float,
        signal_scale: float,
        poisson_fluctuations: bool,
        dim: int = 2
    ):
    '''
    Returns gaussian samples A, B and Sig suitable for fitting.
    
    Parameters
    ----------
    n_ref: number of events in sample A.
    n_bkg: number of events in sample B.
    n_signal: number of events in the added signal.
    noramlization_factor: normalization of the gaussian distribution.
    signal_location: mean of the gaussian distribution of the signal.
    signal_scale: standard deviation of the gaussian distribution of the signal.
    poisson_fluctuations: True - add poisson fluctuations to the number of events in each sample. False - do not add poisson fluctuations.
    dim: dimension of the gaussian distribution.

    Returns
    -------
    reference: numpy array
    background: numpy array
    signal: numpy array
    '''
    n_bkg_pois  = np.random.poisson(lam = n_bkg * np.exp(normalization_factor), size = 1)[0] if poisson_fluctuations else n_bkg
    n_ref_pois  = np.random.poisson(lam = n_ref * np.exp(normalization_factor), size = 1)[0] if poisson_fluctuations else n_ref
    n_Sig_Pois = np.random.poisson(lam = n_signal * np.exp(normalization_factor), size = 1)[0] if poisson_fluctuations else n_signal
    background = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.ones((dim,dim)), size=n_bkg_pois)
    reference  = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.ones((dim,dim)), size=n_ref_pois)
    signal = np.random.multivariate_normal(mean = signal_location * np.ones(dim), cov = signal_scale * np.ones((dim, dim)), size = n_Sig_Pois)
    return reference, background, signal

def physics(
        n_ref: int,
        n_bkg: int,
        n_signal: int,
        channel: str = 'em',
        signal_types: List[str] = ["ggH_taue","vbfH_taue"],
        used_physical_variables: List[str] = ['Mcoll'],
        poisson_fluctuations: bool = True,
        combined_portion: float = 1,
        binned=False,
        resolution=0.05
    ):
    '''
    Turns samples of the selected physical parameters into numpy arrays of samples A, B and Sig suitable for fitting.
    
    Parameters
    ----------
    n_ref: number of events in sample A as a fraction of the total number of events in the sample. (e.g. '1/2','1/10'...)
    n_bkg: number of events in sample B as a fraction of the total number of events in the sample. (e.g. '1/2','1/10'...)
    n_signal: number of total signal events. According to it, selecting randomly from all signal types, with replacement and equal probability)
    channels: decay channel to be used. ('ee','em','me','mm')
    signal_types: new physics signal types to be used. ('ggH_taue','ggH_taumu','vbfH_taue','vbfH_taumu','Z_taue','Z_taumu')
    used_physical_variables: physical variables to be used. ('Mcoll','Lep0Pt','MLL',...)
    poisson_fluctuations: True - add poisson fluctuations to the number of events in each sample. False - do not add poisson fluctuations.
    combined_portion: total number of events (as a fraction) in the over-all sample (before splitting into A and B). (e.g. 1 for ~2e5 events, 0.05 for ~1e4 events)
    binned: True - round the samples to the nearest integer multiple of 'resolution' (if we want to limit ourselves to detectors resolution, for example.). False - do not round the samples.
    resolution: the resolution to which we round the samples (if binned = True).
    
    Returns
    -------
    A: numpy array
    B: numpy array
    Sig: numpy array
    '''
    background = {}
    signal = {}
    vars = used_physical_variables
    # background[f"{channel}_background"] = np.concatenate((np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{var}_dist.npy") for var in vars),axis=1)
    # for s in signal_types:
    #     signal[f"{s}_{channel}_signal"] = np.concatenate((np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{s}_signal_{var}_dist.npy") for var in vars),axis=1)
    # total_Sig = np.concatenate(tuple([signal[f"{s}_{channel}_signal"] for s in sig_types]),axis=0)
    background[f"{channel}_background"] = np.concatenate(tuple([np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{var}_dist.npy") for var in vars]),axis=1)
    for s in signal_types:
        signal[f"{s}_{channel}_signal"] = np.concatenate(tuple([np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{s}_signal_{var}_dist.npy") for var in vars]),axis=1) if n_signal>0 else np.empty((0,background[f"{channel}_background"].shape[1]))
    total_Sig = np.concatenate(tuple([signal[f"{s}_{channel}_signal"] for s in signal_types]),axis=0)

    N_A_Pois  = np.random.poisson(lam=float(Fraction(n_ref))*background[f"{channel}_background"].shape[0]*combined_portion, size=1)[0] if poisson_fluctuations else float(Fraction(n_ref))*background[f"{channel}_background"].shape[0]*combined_portion
    N_B_Pois  = np.random.poisson(lam=float(Fraction(n_bkg))*background[f"{channel}_background"].shape[0]*combined_portion, size=1)[0] if poisson_fluctuations else float(Fraction(n_bkg))*background[f"{channel}_background"].shape[0]*combined_portion
    N_Sig_Pois = np.random.poisson(lam=n_signal, size=1)[0] if poisson_fluctuations else n_signal
    print(N_A_Pois,N_B_Pois,N_Sig_Pois)
    
    # bootstrapping
    # A = np.random.choice(background[f"{channel}_{var}_background"].reshape(-1,),N_A_Pois,replace=True).reshape(-1,1)
    # B = np.random.choice(background[f"{channel}_{var}_background"].reshape(-1,),N_B_Pois,replace=True).reshape(-1,1)
    # Sig = np.random.choice(total_Sig,N_Sig_Pois,replace=True).reshape(-1,1)
    A_events = np.random.choice(np.arange(background[f"{channel}_background"].shape[0]),N_A_Pois,replace=True)
    A = background[f"{channel}_background"][A_events]
    B_events = np.random.choice(np.arange(background[f"{channel}_background"].shape[0]),N_B_Pois,replace=True)
    B = background[f"{channel}_background"][B_events]
    Sig_events = np.random.choice(np.arange(total_Sig.shape[0]),N_Sig_Pois,replace=True)
    Sig = total_Sig[Sig_events]

    if binned:
        A = np.floor(A/resolution)*resolution
        B = np.floor(B/resolution)*resolution
        Sig = np.floor(Sig/resolution)*resolution
    A = normalize(A, normalization)
    B = normalize(B, normalization)
    Sig = normalize(Sig, normalization)
    print(f'setting: {channel}, N_A = {float(Fraction(n_ref))*background["em_background"].shape[0]*combined_portion}, N_B = {float(Fraction(n_bkg))*background["em_background"].shape[0]*combined_portion}, N_Sig = {n_signal}, N_poiss = {poisson_fluctuations}')
    print('A: ',A.shape,' B: ',B.shape, ' Sig: ',Sig.shape)

    return A,B,Sig


def normalize(dataset, normalization=1e5):
    '''
    Normalizes the dataset according to the normalization.
    
    Parameters
    ----------
    normalization : float, str
        if float - the normalization factor. if str ('min-max' or 'standard') - the type of the normalization (incomplete for now).
    '''
    if normalization == 'min-max':
        dataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    elif normalization == 'standard':
        dataset = (dataset - np.mean(dataset)) / np.std(dataset)
    elif isinstance(normalization,float) or isinstance(normalization,int):
        dataset = dataset/normalization
    else:
        raise ValueError("Invalid normalization, see docstring for options")
    return dataset


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
