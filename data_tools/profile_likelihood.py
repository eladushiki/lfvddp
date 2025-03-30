from typing import Union
import h5py,os,glob,numpy as np


def get_names(data,wc,sig,Ref,Bkg,pdf=False,epochs=500000,calc="new",npoiss=""):
    pdf_ext = '/pdf_' if pdf else "/"
    N_poiss = npoiss+"N_poiss" if len(npoiss)>0 else npoiss
    if wc=="9":
        WC = ""
    else:
        WC = f'{wc}BSMweight_clipping'
    delta = "*epochs_delta*patience_delta*" if calc=="new" else ""
    file_name = f"{pdf_ext}{data}{epochs}epochs_tau{delta}{WC}{N_poiss}{sig}signals_{Ref}Ref_{Bkg}Bkg"
    return file_name


def read_weights(filenames):
    dense_biases = {0:[],1:[],2:[],3:[]}
    dense_kernels = {0:[],1:[],2:[],3:[]}
    dense_1_biases = {0:[]}
    dense_1_kernels = {0:[],1:[],2:[],3:[]} 
    for filename in filenames:
        with h5py.File(filename, "r") as f:
            file_items = list(f.items())
            bs_mfinder_net = f.get("bs_mfinder_net")
            bs_mfinder_net_items = list(bs_mfinder_net.items())
            dense = bs_mfinder_net.get("dense")
            dense_items = list(dense.items())
            # 1st biases?
            dense_bias = dense.get("bias:0")
            # 1st weights?
            dense_kernel = dense.get("kernel:0")
            dense_1 = bs_mfinder_net.get("dense_1") 
            dense_1_items = list(dense_1.items())
            # 2nd biases?
            dense_1_bias = dense_1.get("bias:0")
            # 2nd weights?
            dense_1_kernel = dense_1.get("kernel:0")
            dense_1_biases[0].append(np.array(dense_1_bias)[0])
            for i in range(4):
                dense_biases[i].append(np.array(dense_bias)[i])
                dense_kernels[i].append(np.array(dense_kernel)[0][i])
                dense_1_kernels[i].append(np.array(dense_1_kernel)[i][0])
    return dense_kernels,dense_biases,dense_1_kernels,dense_1_biases


def calc_t_test_statistic(tau: Union[int, float, np.ndarray]) -> Union[int, float, np.ndarray]:
    """
    Calculate the test statistic t from the tau value
    """
    return -2 * tau


def read_loss(filenames):
    tau_history = []
    for filename in filenames:      
        with h5py.File(filename, "r") as f:
                epochs_check  = [(key) for key in list(f.keys())]
                t = f.get(str(epochs_check[2]))#'loss'
                t = np.array(t)
                f.close()
                tau_history.append(t)
                #file_items = list(f.items())
    tau_history=np.array(tau_history)
    return tau_history

def read_val_loss(filenames):
    tau_history = []
    for filename in filenames:      
        with h5py.File(filename, "r") as f:
                epochs_check  = [(key) for key in list(f.keys())]
                t = f.get(str('val_loss'))#'loss'
                t = np.array(t)
                f.close()
                tau_history.append(t)
                #file_items = list(f.items())
    tau_history=np.array(tau_history)
    return tau_history


def get_weights(sig,Ref,Bkg,dirInput):
    Ref = Ref.replace('/',':')
    Bkg = Bkg.replace('/',':')
    dirYuval = dirInput#'/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/'
    dir = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
    extract_file_Yuval = dirYuval + f'em500000epochs{sig}signals_{Ref}Ref_{Bkg}Bkg.tar.gz'
    #extract_file= dir + f'em500000epochs{sig}signals_{Ref}Ref_{Bkg}Bkg.tar.gz'
    #os.system(f'cp {extract_file_Yuval} {extract_file}')
    os.system(f'tar --force-local -xzf {extract_file_Yuval} -C {dir}extract_here')
    files = glob.glob(f"{dir}extract_here/*weights*")
    dense_biases = {0:[],1:[],2:[],3:[]}
    dense_kernels = {0:[],1:[],2:[],3:[]}
    dense_1_biases = {0:[]}
    dense_1_kernels = {0:[],1:[],2:[],3:[]} 
    for filename in files:
        with h5py.File(filename, "r") as f:
            file_items = list(f.items())
            bs_mfinder_net = f.get("bs_mfinder_net")
            bs_mfinder_net_items = list(bs_mfinder_net.items())
            dense = bs_mfinder_net.get("dense")
            dense_items = list(dense.items())
            # 1st biases?
            dense_bias = dense.get("bias:0")
            # 1st weights?
            dense_kernel = dense.get("kernel:0")
            dense_1 = bs_mfinder_net.get("dense_1") 
            dense_1_items = list(dense_1.items())
            # 2nd biases?
            dense_1_bias = dense_1.get("bias:0")
            # 2nd weights?
            dense_1_kernel = dense_1.get("kernel:0")
            dense_1_biases[0].append(np.array(dense_1_bias)[0])
            for i in range(4):
                dense_biases[i].append(np.array(dense_bias)[i])
                dense_kernels[i].append(np.array(dense_kernel)[0][i])
                dense_1_kernels[i].append(np.array(dense_1_kernel)[i][0])
    os.system(f'rm -r {dir}extract_here/*')
    return dense_biases,dense_kernels,dense_1_biases,dense_1_kernels


def get_tau_history(sig,Ref,Bkg):
    Ref = Ref.replace('/',':')
    Bkg = Bkg.replace('/',':')
    dirYuval = '/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/'
    dir = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
    extract_file_Yuval = dirYuval + f'em500000epochs{sig}signals_{Ref}Ref_{Bkg}Bkg.tar.gz'
    extract_file= dir + f'em500000epochs{sig}signals_{Ref}Ref_{Bkg}Bkg.tar.gz'
    #os.system(f'cp {extract_file_Yuval} {extract_file}')
    os.system(f'tar --force-local -xzf {extract_file_Yuval} -C {dir}extract_here')
    files = glob.glob(f"{dir}extract_here/*TAU_history*")
    tau_history = []
    for filename in files:
        with h5py.File(filename, "r") as f:
            epochs_check  = [(key) for key in list(f.keys())]
            t = f.get(str(epochs_check[2]))#'loss'
            t = np.array(t)
            f.close()
            tau_history.append(t)
            #file_items = list(f.items())
    tau_history=np.array(tau_history)
    os.system(f'rm -r {dir}extract_here/*')
    return tau_history


def get_final_tau(sig,Ref,Bkg):
    Ref = Ref.replace('/',':')
    Bkg = Bkg.replace('/',':')
    dirYuval = '/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/'
    dir = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
    extract_file_Yuval = dirYuval + f'em500000epochs{sig}signals_{Ref}Ref_{Bkg}Bkg.tar.gz'
    extract_file= dir + f'em500000epochs{sig}signals_{Ref}Ref_{Bkg}Bkg.tar.gz'
    #os.system(f'cp {extract_file_Yuval} {extract_file}')
    os.system(f'tar --force-local -xzf {extract_file_Yuval} -C {dir}extract_here')
    files = glob.glob(f"{dir}extract_here/*TAU_history*")
    final_tau = []
    for filename in files:
        with h5py.File(filename, "r") as f:
            epochs_check  = [(key) for key in list(f.keys())]
            t = f.get(str(epochs_check[2]))#'loss'
            t = np.array(t)
            f.close()
            final_tau.append(t[-1])
            #file_items = list(f.items())
    final_tau=np.array(final_tau)
    os.system(f'rm -r {dir}extract_here/*')
    return final_tau


def get_tau_seed(data,wc,sig,Ref,Bkg,pdf="",epochs=500000,tau = "TAU",calc="new"):
    Ref = Ref.replace('/',':')
    Bkg = Bkg.replace('/',':')
    dirYuval = '/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/'
    dir = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
    extract_file_Yuval = dirYuval + f'em500000epochs_tau{sig}signals_{Ref}Ref_{Bkg}Bkg.tar.gz'
    extract_file= dir + f'em500000epochs{sig}signals_{Ref}Ref_{Bkg}Bkg.tar.gz'
    #os.system(f'cp {extract_file_Yuval} {extract_file}')
    os.system(f'tar --force-local -xzf {extract_file_Yuval} -C {dir}extract_here')
    files = glob.glob(f"{dir}extract_here/*TAU_history*")
    final_tau = []
    seed = []
    start = 'seed'
    end = '_TAU'
    for filename in files:
        seed.append(filename[filename.find(start)+len(start):filename.rfind(end)])
        with h5py.File(filename, "r") as f:
            epochs_check  = [(key) for key in list(f.keys())]
            t = f.get(str(epochs_check[2]))#'loss'
            t = np.array(t)
            f.close()
            final_tau.append(t[-1])
    final_tau=np.array(final_tau)
    seed = np.array(seed)
    os.system(f'rm -r {dir}extract_here/*')
    return final_tau,seed


def sum_tau_delta(file,save=False):
    df = pd.read_csv(file,header=None)
    df_copy = df.copy()
    df_copy[1] = df_copy[1].str.replace("_delta.txt","_TAU+delta.txt")
    df_copy[1] = df_copy[1].str.replace("_TAU.txt","_TAU+delta.txt")
    df_copy = df_copy.groupby(1).sum().reset_index()
    df_copy[[0,1]] = df_copy[[1,0]]
    df_copy.rename(columns={1:0,0:1},inplace=True)
    if save:
        new_df = pd.concat([df,df_copy])
        new_df.to_csv(file,index=False,header=False)
    return df_copy[0].tolist()
    
    
