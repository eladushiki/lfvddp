import numpy as np
import scipy.stats as sps
from sklearn.model_selection import train_test_split
from fractions import Fraction


def prepare_training_valid(data_sets,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,seed,valid,*args, **kwargs):
    '''
    Creates sets of featureData and featureRef according to featureData_str and featureRef_str respectively.
    
    Parameters
    ----------
    data_sets : function
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
    data_sets_dict = {'Ref':np.array([]),'Bkg':np.array([]),'Sig':np.array([]),'':np.array([[]])}
    data_sets_train = {'Ref':np.array([]),'Bkg':np.array([]),'Sig':np.array([]),'':np.array([[]])}
    data_sets_valid = {'Ref':np.array([]),'Bkg':np.array([]),'Sig':np.array([]),'':np.array([[]])}
    data_sets_dict['Ref'],data_sets_dict['Bkg'],data_sets_dict['Sig'] = data_sets(*args, **kwargs)
    print("finish dict")

    

    for key in data_sets_dict.keys():
        try:
            data_sets_train[key],data_sets_valid[key] = train_test_split(data_sets_dict[key],test_size=valid,random_state=seed)
        except:
            data_sets_train[key] = data_sets_dict[key]
            data_sets_valid[key] = data_sets_dict[key]

    Bkg_Aux = np.concatenate(tuple([data_sets_train[key] for key in Bkg_Aux_str.split('+',Bkg_Aux_str.count('+'))]),axis=0).reshape(-1,1)
    Sig_Aux = np.concatenate(tuple([data_sets_train[key] for key in Sig_Aux_str.split('+',Sig_Aux_str.count('+'))]),axis=0).reshape(-1,1)
    Bkg_Data = np.concatenate(tuple([data_sets_train[key] for key in Bkg_Data_str.split('+',Bkg_Data_str.count('+'))]),axis=0).reshape(-1,1)
    Sig_Data = np.concatenate(tuple([data_sets_train[key] for key in Sig_Data_str.split('+',Sig_Data_str.count('+'))]),axis=0).reshape(-1,1)

    
    # Bkg_Aux = np.concatenate(tuple([data_sets_dict[key] for key in train_test_split(Bkg_Aux_str.split('+',Bkg_Aux_str.count('+')),train_size=0.9,test_size=0.1,random_state=seed)[0]]),axis=0).reshape(-1,1)
    # Sig_Aux = np.concatenate(tuple([data_sets_dict[key] for key in train_test_split(Sig_Aux_str.split('+',Sig_Aux_str.count('+')),train_size=0.9,test_size=0.1,random_state=seed)[0]]),axis=0).reshape(-1,1)
    # Bkg_Data = np.concatenate(tuple([data_sets_dict[key] for key in train_test_split(Bkg_Data_str.split('+',Bkg_Data_str.count('+')),train_size=0.9,test_size=0.1,random_state=seed)[0]]),axis=0).reshape(-1,1)
    # Sig_Data = np.concatenate(tuple([data_sets_dict[key] for key in train_test_split(Sig_Data_str.split('+',Sig_Data_str.count('+')),train_size=0.9,test_size=0.1,random_state=seed)[0]]),axis=0).reshape(-1,1)
    # Bkg_Aux_valid = np.concatenate(tuple([data_sets_dict[key] for key in train_test_split(Bkg_Aux_str.split('+',Bkg_Aux_str.count('+')),train_size=0.9,test_size=0.1,random_state=seed)[1]]),axis=0).reshape(-1,1)
    # Sig_Aux_valid = np.concatenate(tuple([data_sets_dict[key] for key in train_test_split(Sig_Aux_str.split('+',Sig_Aux_str.count('+')),train_size=0.9,test_size=0.1,random_state=seed)[1]]),axis=0).reshape(-1,1)
    # Bkg_Data_valid = np.concatenate(tuple([data_sets_dict[key] for key in train_test_split(Bkg_Data_str.split('+',Bkg_Data_str.count('+')),train_size=0.9,test_size=0.1,random_state=seed)[1]]),axis=0).reshape(-1,1)
    # Sig_Data_valid = np.concatenate(tuple([data_sets_dict[key] for key in train_test_split(Sig_Data_str.split('+',Sig_Data_str.count('+')),train_size=0.9,test_size=0.1,random_state=seed)[1]]),axis=0).reshape(-1,1)
   
    N_Bkg = Bkg_Data.shape[0]
    print(N_Bkg)
    featureData = np.concatenate((Bkg_Data,Sig_Data),axis=0)

    featureRef = np.concatenate((Bkg_Aux,Sig_Aux),axis=0)
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


    Bkg_Aux_valid = np.concatenate(tuple([data_sets_valid[key] for key in Bkg_Aux_str.split('+',Bkg_Aux_str.count('+'))]),axis=0).reshape(-1,1)
    Sig_Aux_valid = np.concatenate(tuple([data_sets_valid[key] for key in Sig_Aux_str.split('+',Sig_Aux_str.count('+'))]),axis=0).reshape(-1,1)
    Bkg_Data_valid = np.concatenate(tuple([data_sets_valid[key] for key in Bkg_Data_str.split('+',Bkg_Data_str.count('+'))]),axis=0).reshape(-1,1)
    Sig_Data_valid = np.concatenate(tuple([data_sets_valid[key] for key in Sig_Data_str.split('+',Sig_Data_str.count('+'))]),axis=0).reshape(-1,1)
   
    N_Bkg_valid = Bkg_Data_valid.shape[0]
    print(N_Bkg_valid)
    featureData_valid = np.concatenate((Bkg_Data_valid,Sig_Data_valid),axis=0)

    featureRef_valid = np.concatenate((Bkg_Aux_valid,Sig_Aux_valid),axis=0)
    N_ref_valid = featureRef_valid.shape[0]
    print(N_ref_valid)

    feature_valid     = np.concatenate((featureData_valid, featureRef_valid), axis=0)
    N_R_valid        = N_ref_valid
    N_D_valid        = featureData_valid.shape[0]#N_Bkg

    ## target
    targetData_valid  = np.ones_like(featureData_valid)
    targetRef_valid   = np.zeros_like(featureRef_valid)
    weightsData_valid = np.ones_like(featureData_valid)
    weightsRef_valid  = np.ones_like(featureRef_valid)*N_D_valid*1./N_R_valid
    target_valid     = np.concatenate((targetData_valid, targetRef_valid), axis=0)
    weights_valid     = np.concatenate((weightsData_valid, weightsRef_valid), axis=0)
    target_valid      = np.concatenate((target_valid, weights_valid), axis=1)
    
    return feature,target,feature_valid,target_valid

def prepare_training(data_sets,Bkg_Aux_str,Sig_Aux_str,Bkg_Data_str,Sig_Data_str,*args, **kwargs):
    '''
    Creates sets of featureData and featureRef according to featureData_str and featureRef_str respectively.
    
    Parameters
    ----------
    data_sets : function
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
    data_sets_dict = {'Ref':np.array([]),'Bkg':np.array([]),'Sig':np.array([]),'':np.array([[]])}
    data_sets_dict['Ref'],data_sets_dict['Bkg'],data_sets_dict['Sig'] = data_sets(*args, **kwargs)
    print("finish dict")
    Bkg_Aux = np.concatenate(tuple([data_sets_dict[key] for key in Bkg_Aux_str.split('+',Bkg_Aux_str.count('+'))]),axis=0).reshape(-1,1)
    Sig_Aux = np.concatenate(tuple([data_sets_dict[key] for key in Sig_Aux_str.split('+',Sig_Aux_str.count('+'))]),axis=0).reshape(-1,1)
    Bkg_Data = np.concatenate(tuple([data_sets_dict[key] for key in Bkg_Data_str.split('+',Bkg_Data_str.count('+'))]),axis=0).reshape(-1,1)
    Sig_Data = np.concatenate(tuple([data_sets_dict[key] for key in Sig_Data_str.split('+',Sig_Data_str.count('+'))]),axis=0).reshape(-1,1)
   
    N_Bkg = Bkg_Data.shape[0]
    print(N_Bkg)
    featureData = np.concatenate((Bkg_Data,Sig_Data),axis=0)

    featureRef = np.concatenate((Bkg_Aux,Sig_Aux),axis=0)
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


def exp(N_Ref,N_Bkg,N_Sig,seed,Scale=0,Norm=0,Sig_loc=6.4,Sig_scale=0.16, N_poiss = False):
    np.random.seed(seed)

    N_Bkg_Pois  = np.random.poisson(lam=N_Bkg*np.exp(Norm), size=1)[0]if N_poiss else N_Bkg
    N_Ref_Pois  = np.random.poisson(lam=N_Ref*np.exp(Norm), size=1)[0] if N_poiss else N_Ref
    N_Sig_Pois = np.random.poisson(lam=N_Sig*np.exp(Norm), size=1)[0] if N_poiss else N_Sig

    Bkg = np.random.exponential(scale=np.exp(1*Scale), size=(N_Bkg_Pois, 1))
    Ref  = np.random.exponential(scale=1., size=(N_Ref_Pois, 1))
    Sig = np.random.normal(loc=Sig_loc, scale=Sig_scale, size=(N_Sig_Pois,1))*np.exp(Scale)

    return Ref,Bkg,Sig


def em(train_size,test_size,sig_events,seed,sample_pdf=False,size_sample_pdf=219087,Sig_pdf=False,size_Sig_pdf=5000):
    np.random.seed(seed)
    channel='em'
    signal_samples=["ggH_taue","ggH_taumu","vbfH_taue","vbfH_taumu","Z_taue","Z_taumu"]
    background = {}
    signal = {}
    background["%s_background"%channel] = np.load("/storage/agrp/yuvalzu/NPLM/%s_MLL_dist.npy"%channel)
    for s in signal_samples:
        signal["%s_%s_signal"%(s,channel)] = np.load("/storage/agrp/yuvalzu/NPLM/%s_%s_signal_MLL_dist.npy"%(channel,s))

    
    data_dist = generate_pdf(background["em_background"],seed,size=size_sample_pdf) if sample_pdf else background["em_background"]
    Ref,Bkg = train_test_split(data_dist, train_size=float(Fraction(train_size)), test_size=float(Fraction(test_size)),random_state=seed)
    
    if Sig_pdf:
        Sig_dist = generate_pdf(signal["ggH_taue_em_signal"],seed,size=size_Sig_pdf)
    else:
        if sig_events<len(signal["ggH_taue_em_signal"].reshape(-1,)):
            Sig_dist = signal["ggH_taue_em_signal"].reshape(-1,)
        else:
            Sig_dist = np.tile(signal["ggH_taue_em_signal"],10).reshape(-1,)
    Sig = np.random.choice(Sig_dist,sig_events,False).reshape(-1,1)

    return Ref/1e5,Bkg/1e5,Sig/1e5


def generate_pdf(x,seed,size=3e6):
    '''
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