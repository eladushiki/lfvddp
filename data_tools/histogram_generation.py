import numpy as np
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from fractions import Fraction


def prepare_training(datasets,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,*args, **kwargs):
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
    print("creating dict")
    datasets_dict = {'Ref':np.array([]),'Bkg':np.array([]),'Sig':np.array([]),'':np.array([[]])}
    datasets_dict['Ref'],datasets_dict['Bkg'],datasets_dict['Sig'] = datasets(*args, **kwargs)
    print("finish dict")
    Bkg_Aux = np.concatenate(tuple([datasets_dict[key] for key in Bkg_Aux_str.split('+',Bkg_Aux_str.count('+'))]),axis=0)
    Sig_Aux = np.concatenate(tuple([datasets_dict[key] for key in Sig_Aux_str.split('+',Sig_Aux_str.count('+'))]),axis=0) if Sig_Aux_str!='' else np.zeros((0,Bkg_Aux.shape[1]))
    Bkg_Data = np.concatenate(tuple([datasets_dict[key] for key in Bkg_Data_str.split('+',Bkg_Data_str.count('+'))]),axis=0)
    Sig_Data = np.concatenate(tuple([datasets_dict[key] for key in Sig_Data_str.split('+',Sig_Data_str.count('+'))]),axis=0) if Sig_Data_str!='' else np.zeros((0,Bkg_Data.shape[1]))
   
    N_Bkg = Bkg_Data.shape[0]
    print(N_Bkg)
    featureData = np.concatenate((Bkg_Data,Sig_Data),axis=0)

    featureRef = np.concatenate((Bkg_Aux,Sig_Aux),axis=0)
    N_ref = featureRef.shape[0]
    print(N_ref)

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


def resample(feature,target,resample_seed,Bkg_Data_str = "Bkg",label_method="permute", N_method = "fixed", replacement = True):
    '''
    Creates sets of featureData and featureRef according to featureData_str and featureRef_str respectively.
    
    Parameters
    ----------
    label_method : str = "permute"|"binomial"|"bootstrap" 
        "permute": train_test_split according to current ratio (number of events in each sample remains unchanged)
        "binomial": determine label according to binomial distribution with p_A = N_A/(N_A+N_B) - only total number of events is fixed
        "bootstrap": resample uniformly (with/wo replacement) with sizes N_A and N_B 
    
    N_method : str = "fixed"|"binomial"
    N_A, N_B : mean number of events 

    replacement : for bootstrap only - whether events can be repeated. True = repeat events, False - do not repeat events.

    returns feature and target
    '''
    combined_data = feature[target[:, 0]==0]
    N_combined = combined_data.shape[0]
    np.random.seed(resample_seed)
    if "Ref" in Bkg_Data_str:
        N_B = feature[target[:, 0]==1].shape[0]
        N_A = N_combined-N_B
        print("Bkg")
    else:
        N_A = feature[target[:, 0]==1].shape[0]
        N_B = N_combined-N_A
        print("Bkg")
    
    if N_method=="poiss":
        N_A = np.random.poisson(lam=N_A, size=1)[0]
        N_B = np.random.poisson(lam=N_B, size=1)[0]

    print(f"N_A = {N_A}, N_B = {N_B}")
    
    if label_method=="permute":
        data_A, data_B = train_test_split(combined_data,train_size=N_A,test_size = N_B,random_state=resample_seed)
    elif label_method =="binomial":
        np.random.seed(resample_seed)
        label = np.random.choices(["A","B"],weights = [N_A/(N_A+N_B),N_B/(N_A+N_B)],k = N_combined)
        data_A = combined_data[label=="A"]
        data_B = combined_data[label=="B"]
    elif label_method=="bootstrap":
        np.random.seed(resample_seed)
        data_A = np.random.choices(combined_data, replace = replacement, k=N_A)
        np.random.seed(resample_seed+1)
        data_B = np.random.choices(combined_data, replace = replacement, k=N_B)
    else:
        print("no resampling")
        return feature,target

    if "Ref" in Bkg_Data_str:
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



def exp(N_A,N_B,N_Sig,seed,Scale=0,Norm=0,Sig_loc=6.4,Sig_scale=0.16,N_poiss=True,resonant=True):
    '''
    Returns exponential samples A, B and Sig suitable for fitting.
    
    Parameters
    ----------
    N_A : int
        number of events in sample A.
    N_B : int
        number of events in sample B.
    N_Sig : int
        number of events in sample Sig.
    seed : int
        seed for random functions.
    Scale : float
        scale of the exponential distribution.
    Norm : float
        normalization of the exponential distribution.
    Sig_loc : float
        mean of the gaussian distribution of the signal.
    Sig_scale : float
        standard deviation of the gaussian distribution of the signal.
    N_poiss : bool
        True - add poisson fluctuations to the number of events in each sample. False - do not add poisson fluctuations.
    resonant : bool
        True - add gaussian signal. False - add non-resonant signal.

    Returns
    -------
    A : numpy array
    B : numpy array
    Sig : numpy array
    '''
    np.random.seed(seed)

    N_Bkg_Pois  = np.random.poisson(lam=N_Bkg*np.exp(Norm), size=1)[0] if N_poiss else N_Bkg
    N_Ref_Pois  = np.random.poisson(lam=N_Ref*np.exp(Norm), size=1)[0] if N_poiss else N_Ref
    N_Sig_Pois = np.random.poisson(lam=N_Sig*np.exp(Norm), size=1)[0] if N_poiss else N_Sig
    print(N_Bkg_Pois,N_Ref_Pois,N_Sig_Pois)

    Bkg = np.random.exponential(scale=np.exp(1*Scale), size=(N_Bkg_Pois, 1))
    Ref  = np.random.exponential(scale=1., size=(N_Ref_Pois, 1))
    if resonant:
        Sig = np.random.normal(loc=Sig_loc, scale=Sig_scale, size=(N_Sig_Pois,1))*np.exp(Scale)
    if not resonant:
        def Sig_dist(x):
            dist = x**2*np.exp(-x)
            return dist/np.sum(dist)
        Sig = np.random.choice(np.linspace(0,100,100000),size=(N_Sig_Pois,1),replace=True,p=Sig_dist(np.linspace(0,100,100000)))*np.exp(Scale)   
    print(f'defs: exp, N_Ref={N_A},N_Bkg={N_B},N_Sig={N_Sig},seed = {seed},Scale={Scale},Norm={Norm},Sig_loc={Sig_loc},Sig_scale={Sig_scale}, N_poiss = {N_poiss}, resonant = {resonant}')
    print('Ref',A.shape,'Bkg',B.shape, 'Sig',Sig.shape)
    return A,B,Sig


def gauss(
        N_Ref: int,
        N_Bkg: int,
        N_Sig: int,
        Norm: float,
        Sig_loc: float,
        Sig_scale: float,
        N_poiss: bool,
        dim=2):
    '''
    Returns gaussian samples A, B and Sig suitable for fitting.
    
    Parameters
    ----------
    N_Ref: number of events in sample A.
    N_Bkg: number of events in sample B.
    N_Sig: number of events in sample Sig.
    Norm: normalization of the gaussian distribution.
    Sig_loc: mean of the gaussian distribution of the signal.
    Sig_scale: standard deviation of the gaussian distribution of the signal.
    N_poiss: True - add poisson fluctuations to the number of events in each sample. False - do not add poisson fluctuations.
    resonant: True - add gaussian signal. False - add non-resonant signal.
    dim: dimension of the gaussian distribution.

    Returns
    -------
    reference: numpy array
    background: numpy array
    signal: numpy array
    '''
    n_bkg_pois  = np.random.poisson(lam = N_Bkg * np.exp(Norm), size = 1)[0] if N_poiss else N_Bkg
    n_ref_pois  = np.random.poisson(lam = N_Ref * np.exp(Norm), size = 1)[0] if N_poiss else N_Ref
    n_Sig_Pois = np.random.poisson(lam = N_Sig * np.exp(Norm), size = 1)[0] if N_poiss else N_Sig
    background = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.ones((dim,dim)), size=n_bkg_pois)
    reference  = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.ones((dim,dim)), size=n_ref_pois)
    signal = np.random.multivariate_normal(mean = Sig_loc * np.ones(dim), cov = Sig_scale * np.ones((dim, dim)), size = n_Sig_Pois)
    return reference, background, signal


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

def physics(A_size, B_size, sig_events, seed, channels=['em'], signal_types=["ggH_taue","vbfH_taue"], phys_variables=['Mcoll'], N_poiss=True ,combined_portion=1, binned=False, resolution=0.05):
    '''
    Turns samples of the selected physical parameters into numpy arrays of samples A, B and Sig suitable for fitting.
    
    Parameters
    ----------
    A_size : str
        number of events in sample A as a fraction of the total number of events in the sample. (e.g. '1/2','1/10'...)
    B_size : str
        number of events in sample B as a fraction of the total number of events in the sample. (e.g. '1/2','1/10'...)
    sig_events : int
        number of total signal events. According to it, selecting randomly from all signal types, with replacement and equal probability)
    seed : int
        seed for random functions.
    channels : str
        decay channel to be used. ('ee','em','me','mm')
    signal_types : list of str
        new physics signal types to be used. ('ggH_taue','ggH_taumu','vbfH_taue','vbfH_taumu','Z_taue','Z_taumu')
    phys_variables : list of str
        physical variables to be used. ('Mcoll','Lep0Pt','MLL',...)
    N_poiss : bool
        True - add poisson fluctuations to the number of events in each sample. False - do not add poisson fluctuations.
    combined_portion : float
        total number of events (as a fraction) in the over-all sample (before splitting into A and B). (e.g. 1 for ~2e5 events, 0.05 for ~1e4 events)
    binned : bool
        True - round the samples to the nearest integer multiple of 'resolution' (if we want to limit ourselves to detectors resolution, for example.). False - do not round the samples.
    resolution : float
        the resolution to which we round the samples (if binned = True).
    
    Returns
    -------
    A : numpy array
    B : numpy array
    Sig : numpy array
    '''
    background = {}
    signal = {}
    vars = phys_variables
    # background[f"{channel}_background"] = np.concatenate((np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{var}_dist.npy") for var in vars),axis=1)
    # for s in signal_types:
    #     signal[f"{s}_{channel}_signal"] = np.concatenate((np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{s}_signal_{var}_dist.npy") for var in vars),axis=1)
    # total_Sig = np.concatenate(tuple([signal[f"{s}_{channel}_signal"] for s in sig_types]),axis=0)
    background[f"{channel}_background"] = np.concatenate(tuple([np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{var}_dist.npy") for var in vars]),axis=1)
    for s in signal_types:
        signal[f"{s}_{channel}_signal"] = np.concatenate(tuple([np.load(f"/storage/agrp/yuvalzu/NPLM/{channel}_{s}_signal_{var}_dist.npy") for var in vars]),axis=1) if sig_events>0 else np.empty((0,background[f"{channel}_background"].shape[1]))
    total_Sig = np.concatenate(tuple([signal[f"{s}_{channel}_signal"] for s in signal_types]),axis=0)

    np.random.seed(seed)
    N_A_Pois  = np.random.poisson(lam=float(Fraction(A_size))*background[f"{channel}_background"].shape[0]*combined_portion, size=1)[0] if N_poiss else float(Fraction(A_size))*background[f"{channel}_background"].shape[0]*combined_portion
    np.random.seed(seed+1)
    N_B_Pois  = np.random.poisson(lam=float(Fraction(B_size))*background[f"{channel}_background"].shape[0]*combined_portion, size=1)[0] if N_poiss else float(Fraction(B_size))*background[f"{channel}_background"].shape[0]*combined_portion
    np.random.seed(seed+2)
    N_Sig_Pois = np.random.poisson(lam=sig_events, size=1)[0] if N_poiss else sig_events
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
    print(f'setting: {channel}, N_A = {float(Fraction(A_size))*background["em_background"].shape[0]*combined_portion}, N_B = {float(Fraction(B_size))*background["em_background"].shape[0]*combined_portion}, N_Sig = {sig_events}, seed = {seed}, N_poiss = {N_poiss}')
    print('A: ',A.shape,' B: ',B.shape, ' Sig: ',Sig.shape)

    return A,B,Sig


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
