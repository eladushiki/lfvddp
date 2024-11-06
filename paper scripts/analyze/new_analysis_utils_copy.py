import numpy as np, os, re, glob, h5py, sys, os, time #, pandas as pd
from scipy.stats import chi2
import scipy.special as spc
from scipy.integrate import quad

sys.path.insert(0,"/storage/agrp/yuvalzu/mattiasdata")
import save_jobs_script as jobs
import utils as u

#------------------------------------------------------------------------------------------------------------------------#

def user_dir():
    dirYuval = "/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/"
    dirInbar = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
    user = jobs.cmdline('whoami')[0][2:-3]
    try:
        if user=="yuvalzu":
            dir = dirYuval
            # plots_dir =  plots_dirYuval
        if user=="inbarsav":
            dir = dirInbar
            # plots_dir =  plots_dirInbar
    except:
        print("invalid user")
    return(dir)

def t_hist_epoch(epochs_list,t_history,epoch_numbers):
    t_hist_dict ={}
    for epoch in epoch_numbers:
        t_hist_dict[epoch] = []
        tepoch = [t_history[i][np.where(epochs_list[i]== epoch)[0]] for i in range(len(epochs_list))]
        if len(tepoch)>0:
            t_hist_dict[epoch] = np.concatenate(tepoch).ravel()
    return t_hist_dict 

def t_hist_epoch_seeds(epochs_list,t_history,seeds_list,epoch_numbers):
    t_hist_dict ={}
    seeds_dict = {}
    for epoch in epoch_numbers:
        t_hist_dict[epoch] = []
        seeds_dict[epoch] = []
        tepoch = [t_history[i][np.where(epochs_list[i]== epoch)[0]] for i in range(len(epochs_list))]
        seeds_epoch = [seeds_list[i][np.where(epochs_list[i]== epoch)[0]] for i in range(len(epochs_list))]
        if len(tepoch)>0:
            t_hist_dict[epoch] = np.concatenate(tepoch).ravel()
            seeds_dict[epoch] = np.concatenate(seeds_epoch).ravel()
    return t_hist_dict, seeds_dict

class results:
    N = 219087
    dir = user_dir()  
    dirYuval = "/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/"
    dirInbar = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
    def __init__(self,file_name):  
        self.file = file_name
        self.tar_file_name = file_name.replace(".csv",".tar.gz") if file_name.endswith(".csv") else file_name
        self.csv_file_name = file_name.replace(".tar.gz",".csv") if file_name.endswith(".tar.gz") else file_name
        self.total_events = float(re.search(r'\d.\d+combined', file_name)[0][:-len('combined')])*results.N if 'combined' in file_name else results.N
        total_factor =  float(re.search(r'\d.\d+combined', file_name)[0][:-len('combined')]) if 'combined' in file_name else 1.0
        bkg_str = re.search(r'\d+:?\d*Bkg', file_name)[0][:-len('Bkg')].split(':')
        self.Bkg_ratio = total_factor*float(bkg_str[0])/float(bkg_str[1]) if len(bkg_str)>1 else total_factor*float(bkg_str[0])
        self.Bkg_events = int(results.N*self.Bkg_ratio)
        ref_str = re.search(r'\d+:?\d*Ref', file_name)[0][:-len('Ref')].split(':')
        self.Ref_ratio = total_factor*float(ref_str[0])/float(ref_str[1]) if len(ref_str)>1 else total_factor*float(ref_str[0])
        self.Ref_events = int(results.N*self.Ref_ratio)
        self.Sig_events = int(re.search(r'\d+signals', file_name)[0][:-len('signals')])
        self.Bkg_sample = "exp" if "exp" in file_name else "em"
        wc = 9
        if "BSMweight_clipping" in file_name:
            if "NoneBSMweight_clipping" in file_name:
                wc = "None"
            else:
                wc = float(re.search(r'\d+.?\d*BSMweight_clipping', file_name)[0][:-len('BSMweight_clipping')])
        self.WC = wc 
        self.N_poiss = "True" if "TrueN_poiss" in file_name else "False"

        self.Sig_resonant = "False" if "Falseresonan" in file_name else "True"
        self.Sig_loc = float(re.search(r'\d+.?\d*Sig_loc', file_name)[0][:-len('Sig_loc')]) if 'Sig_loc' in file_name else 6.4
        self.Sig_scale = float(re.search(r'\d+.?\d*Sig_scale', file_name)[0][:-len('Sig_scale')]) if 'Sig_scale' in file_name else 0.16

        self.resample = "True" if "Trueresample" in file_name else "False"
        if self.resample=="True":
            self.label_method = re.search(r'sample\S+label', file_name)[0][len('sample'):-len('label')]
            self.N_method = re.search(r'method\S+N', file_name)[0][len('method'):-len('N')]
            self.replacement = "True" if "Truereplacement" in file_name else "False"
            self.original_seed = int(re.search(r'\d+original_seed', file_name)[0][:-len('original_seed')])
        else:
            self.label_method = ""
            self.N_method = ""
            self.replacement = ""
            self.original_seed = None

        self.tot_epochs = float(re.search(r'\d+epochs_tau',file_name)[0][:-len('epochs_tau')]) if 'epochs_tau' in file_name else 300000

        
     

    def get_similar_files(self,epochs='all',patience_tau='all',patience_delta='all'):
        dirYuval = "/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/"
        dirInbar = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
        all_patience_str = self.file
        sub_epochs = '*' if epochs=='all' else f'{epochs}epochs_tau'
        sub_patience_tau = '*' if patience_tau=='all' else f'{patience_tau}patience_tau'
        sub_patience_delta = '*' if patience_delta=='all' else f'{patience_delta}patience_delta' 
        all_patience_str = re.sub(r'\d+epochs_delta','*',all_patience_str)
        all_patience_str = re.sub(r'\d+epochs_tau',sub_epochs,all_patience_str)
        all_patience_str = re.sub(r'\d+patience_delta',sub_patience_delta,all_patience_str)
        all_patience_str = re.sub(r'\d+patience_tau',sub_patience_tau,all_patience_str)
        all_patience_str = "exp"+all_patience_str.split('exp')[1] if "exp" in self.file else "em"+all_patience_str.split('em')[1]
        all_patience_str =re.sub('\*\*+', '*', all_patience_str)
        self.similar_search_name = all_patience_str
        files_all_patience_str = glob.glob(dirYuval+all_patience_str)+glob.glob(dirInbar+all_patience_str)
        self.similar_files = files_all_patience_str
        return files_all_patience_str
    
    
    def read_final_t_csv(self):
        TAU_names = []
        TAUs = []
        delta_names = []
        deltas = []
        for file in self.get_similar_files():
            csv_file = file
            with open(csv_file,'r') as f:
                lines = f.readlines()
                TAU_names += [tau.split(',')[1] for tau in lines if tau.count('TAU.')]
                TAUs += [float(tau.split(',')[0]) for tau in lines if tau.count('TAU.')]
                # TAUs = np.array([float(tau.split(',')[0]) for tau in lines if tau.count('TAU.')])
                delta_names += [delta.split(',')[1] for delta in lines if delta.count('delta.')]
                deltas += [float(delta.split(',')[0]) for delta in lines if delta.count('delta.')]
                # deltas = np.array([float(delta.split(',')[0]) for delta in lines if delta.count('delta.')])
        TAU_plus_delta = np.array([TAUs[TAU_names.index(delta_name.replace("delta.txt","TAU.txt"))]+deltas[delta_names.index(delta_name)] for delta_name in delta_names if TAU_names.count(delta_name.replace("delta.txt","TAU.txt"))>0])
        TAU_plus_delta_names = [TAU_names[TAU_names.index(delta_name.replace("delta.txt","TAU.txt"))]+' + '+delta_names[delta_names.index(delta_name)] for delta_name in delta_names if TAU_names.count(delta_name.replace("delta.txt","TAU.txt"))>0]
        delta_names = [delta_names[delta_names.index(delta_name)] for delta_name in delta_names if TAU_names.count(delta_name.replace("delta.txt","TAU.txt"))>0]

        return TAU_plus_delta, delta_names
            
    
    def get_t_history(self):
        dir = results.dir
        similar_files= self.get_similar_files()
        #file_search_name = self.similar_search_name
        for file in similar_files:
            if "tar.gz" not in file:
                file = file.replace("csv","tar.gz") if ".csv" in file else file+".tar.gz"
            os.system(f'tar --force-local -xzf {file} -C {dir}extract_here')
        tar_file = self.tar_file_name
        #csv_file = self.csv_file_name
        t_final,txt_names=self.read_final_t_csv()
        txt_names = [(name.split("/")[-1]).replace("\n","") for name in txt_names]
        #os.system(f'tar --force-local -xzf {dir+tar_file} -C {dir}extract_here')
        history_files = f'{dir}extract_here/*{tar_file.replace(".tar.gz","")}*_history*'
        files = glob.glob(re.sub('\*\*+', '*',history_files))
        t_history = []
        epochs = []
        seeds = []
        if len(files)>0:
            for filename in files:
                patience_tau = float((re.search(r'\d+patience_tau',filename).group()).split("patience")[0] if "patience_tau" in filename else 1000) 
                patience_delta = float((re.search(r'\d+patience_delta',filename).group()).split("patience")[0] if "patience_delta" in filename else 1000)
                patience = max(patience_tau,patience_delta)
                step_tau = round(patience/patience_tau)
                step_delta = round(patience/patience_delta)
                if (('_TAU_history' in filename) and (filename.replace('_TAU_history','_delta_history') in files)):
                    with h5py.File(filename, "r") as f1:
                        keys_list  = [(key) for key in list(f1.keys())]
                        TAU_history = f1.get(str(keys_list[2]))#'loss'
                        TAU_history = np.array(TAU_history)
                    with h5py.File(filename.replace('_TAU_history','_delta_history'), "r") as f2:
                        keys_list  = [(key) for key in list(f2.keys())]
                        delta_history = f2.get(str(keys_list[2]))#'loss'
                        delta_history = np.array(delta_history)
                    seed_num = int(re.search(r'_seed\d+_',filename)[0][len('_seed'):-len('_')])
                    t_history.append(-2*(TAU_history[0::step_tau]+delta_history[0::step_delta]))
                    epochs.append(patience*np.array(range(len(TAU_history[0::step_tau]))))
                    seeds.append(seed_num*np.ones_like(np.array(range(len(TAU_history[0::step_tau])))))
        #os.system(f'rm {dir}extract_here/*')
        if len(t_final)>0:
            for i,t in enumerate(t_final):
                    name = txt_names[i]
                    seed_num = int(re.search(r'_seed\d+_',name)[0][len('_seed'):-len('_')])
                    tot_epochs = float(re.search(r'\d+epochs_tau',name)[0][:-len('epochs_tau')])
                    t_history.append(np.array([t]))
                    epochs.append(np.array([tot_epochs]))
                    seeds.append(seed_num*np.ones_like(np.array([tot_epochs])))
        os.system(f'rm -r {dir}extract_here')
        os.system(f"mkdir {dir}extract_here")
        return t_history,epochs,seeds
    

    def get_t_history_dict(self):
        t_sig_hist, epochs_sig_list, seeds_list = self.get_t_history()#get_history_t_values(self.file)
        epochs_list = np.unique(np.concatenate(epochs_sig_list).ravel())
        Sig_t = t_hist_epoch(epochs_sig_list, t_sig_hist,epochs_list)
        self.t_history = Sig_t
        return Sig_t
    
    def get_signal_files(self,N_sig='all',Sig_loc = 'all',Sig_scale = 'all',resonant="all"):
        filenames = []
        similar_filenames = self.get_similar_files()
        bkg_search_filename = self.similar_search_name.replace("tar.gz","csv")
        #bkg_filename = self.csv_file_name.split('/')[-1] if '/' in self.csv_file_name else self.csv_file_name
        bkg_search_filename =(bkg_search_filename.split('clipping')[0]+'*'+bkg_search_filename.split('signals_')[1])
        sig_filename = re.sub('\*\*+', '*', bkg_search_filename)
        #print(sig_filename)
        sig_files = glob.glob(results.dirYuval+sig_filename)+glob.glob(results.dirInbar+sig_filename)
        #print(sig_files)
        for file in sig_files:
            file = file.split('/')[-1]
            params_file = results(file)
            flag = False
            if (params_file.Bkg_events==self.Bkg_events) and (params_file.Ref_events==self.Ref_events) and (params_file.N_poiss==self.N_poiss):
                flag = True
                if N_sig!="all":
                    flag = flag and (params_file.Sig_events in N_sig)
                if params_file.Sig_events!=0:
                    if Sig_loc!="all":
                        flag = flag and (params_file.Sig_loc in Sig_loc)
                    if Sig_scale!="all":
                        flag = flag and (params_file.Sig_scale in Sig_scale)
                    if resonant!="all":
                        flag = flag and (params_file.Sig_resonant in resonant)
                if flag and (file not in filenames):
                    filenames.append(file)
        return filenames



class exp_results(results):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.Sig_resonant = "False" if "Falseresonan" in file_name else "True"
        self.Sig_loc = float(re.search(r'\d+.?\d*Sig_loc', file_name)[0][:-len('Sig_loc')]) if 'Sig_loc' in file_name else 6.4
        self.Sig_scale = float(re.search(r'\d+.?\d*Sig_scale', file_name)[0][:-len('Sig_scale')]) if 'Sig_scale' in file_name else 0.16

    def get_sqrt_q0(self):
        Bkg_pdf = lambda x: np.exp(-x)
        if 'Falseresonant' in self.file:
            Sig_pdf = lambda x: x**2*np.exp(-x)/2
            integrand = lambda x: (self.Sig_events*Sig_pdf(x)+self.Bkg_events*Bkg_pdf(x))*np.log(1+self.Sig_events*Sig_pdf(x)/(self.Bkg_events*Bkg_pdf(x)))
            sqrt_q0 = -2*(self.Sig_events-(quad(integrand, 0, 200)[0]))#+quad(integrand, 200, np.inf)[0])))
        else:
            Sig_loc = self.Sig_loc
            sigma = self.Sig_scale
            Sig_pdf = lambda x: (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x-Sig_loc)**2/(2*sigma**2))
            integrand = lambda x: (self.Sig_events*Sig_pdf(x)+self.Bkg_events*Bkg_pdf(x))*np.log(1+self.Sig_events*Sig_pdf(x)/(self.Bkg_events*Bkg_pdf(x)))
            sqrt_q0 = -2*(self.Sig_events-(quad(integrand, 0, Sig_loc)[0]+quad(integrand, Sig_loc, np.inf)[0]))
        return np.sqrt(sqrt_q0)
    

    def get_signals_files(self):
        filenames1 =self.get_signal_files(Sig_loc = [6.4],Sig_scale = [0.16],resonant=["True"])
        filenames2 =self.get_signal_files(resonant=["False"])
        filenames3 =self.get_signal_files(Sig_loc = [1.6],Sig_scale = [0.16],resonant=["True"])
        return filenames1, filenames2, filenames3



class em_results(results):
    # channel='em'
    # signal_samples=["ggH_taue","vbfH_taue"]#["ggH_taue","ggH_taumu","vbfH_taue","vbfH_taumu","Z_taue","Z_taumu"]
    # background = {}
    # signal = {}
    # background["em_background"] = np.load("/storage/agrp/yuvalzu/NPLM/em_MLL_dist.npy")
    # for s in signal_samples:
    #     signal[f"{s}_em_signal"] = np.load(f"/storage/agrp/yuvalzu/NPLM/em_{s}_signal_MLL_dist.npy")
    # Bkg = background["em_background"]
    
    def __init__(self, file_name):
        super().__init__(file_name)

    def collect_data(self):
        if 'em_Mcoll' in self.file:
            channel='em'
            signal_samples=["ggH_taue","vbfH_taue"]#["ggH_taue","ggH_taumu","vbfH_taue","vbfH_taumu","Z_taue","Z_taumu"]
            background = {}
            signal = {}
            background["em_background"] = np.load("/storage/agrp/yuvalzu/NPLM/em_Mcoll_dist.npy")
            for s in signal_samples:
                signal[f"{s}_em_signal"] = np.load(f"/storage/agrp/yuvalzu/NPLM/em_{s}_signal_Mcoll_dist.npy")
            Bkg = background["em_background"]
        else:       
            channel='em'
            signal_samples=["ggH_taue","vbfH_taue"]#["ggH_taue","ggH_taumu","vbfH_taue","vbfH_taumu","Z_taue","Z_taumu"]
            background = {}
            signal = {}
            background["em_background"] = np.load("/storage/agrp/yuvalzu/NPLM/em_MLL_dist.npy")
            for s in signal_samples:
                signal[f"{s}_em_signal"] = np.load(f"/storage/agrp/yuvalzu/NPLM/em_{s}_signal_MLL_dist.npy")
            Bkg = background["em_background"]
        return Bkg, signal, signal_samples

    def get_sqrt_q0(self):
        # Bkg = em_results.Bkg
        # signal_samples = em_results.signal_samples
        # signal = em_results.signal
        Bkg, signal, signal_samples = self.collect_data()
        Sig = np.concatenate(tuple([signal[f"{s}_em_signal"] for s in signal_samples]),axis=0).reshape(-1,)
        mu = self.Sig_events/len(Sig)
        bkgFrac = self.Bkg_events/len(Bkg)
        histData = np.histogram(Sig,200)
        bins = histData[1]
        data = histData[0][histData[0]!=0]
        histBkg = np.histogram(Bkg,bins)
        bkg= histBkg[0][histData[0]!=0]
        sqrt_q0 =2*(-self.Sig_events+np.sum((mu*data+(bkg*bkgFrac))*np.log(mu*data/(bkg*bkgFrac)+1)))
        return np.sqrt(sqrt_q0)
    
    def get_signals_files(self):
        filenames =self.get_signal_files(Sig_loc = [6.4],Sig_scale = [0.16],resonant=["True"])
        return filenames
    # def get_signal_files(self):
    #     filenames = []
    #     sig_filename = self.csv_file_name.split('clipping')[0]+'*'+self.csv_file_name.split('signals_')[1]
    #     sig_files = glob.glob(results.dirYuval+sig_filename)+glob.glob(results.dirInbar+sig_filename)
    #     for file in sig_files:
    #         file = file.split('/')[-1]
    #         params_file = results(file)
    #         if (params_file.Bkg_events==self.Bkg_events) and (params_file.Ref_events==self.Ref_events) and (params_file.N_poiss==self.N_poiss):
    #             if file not in filenames:
    #                 filenames.append(file)
    #     return filenames
    


def get_z_score(results_file:results,bkg_results_file:results, epoch = 500000):
    sig_results = results_file
    sig_file_name = sig_results.csv_file_name
    bkg_file_name = bkg_results_file.csv_file_name
    if sig_results.Bkg_sample=="exp":
        bkg_file = exp_results(bkg_file_name)
    elif sig_results.Bkg_sample=="em":
        bkg_file = em_results(bkg_file_name)
    if epoch >= sig_results.tot_epochs:
        sig_t = sig_results.read_final_t_csv()[0]
    else:
        sig_t_dict = sig_results.get_t_history_dict
        sig_t = sig_t_dict[epoch] if epoch in sig_t_dict.keys() else sig_t_dict[sig_results.tot_epochs]
    
    #bkg_t = bkg_file.read_final_t_csv()[0]

    if epoch >= bkg_file.tot_epochs:
        bkg_t = bkg_file.read_final_t_csv()[0]
    else:
        bkg_t_dict = bkg_file.get_t_history_dict
        bkg_t = bkg_t_dict[epoch] if epoch in bkg_t_dict.keys() else bkg_t_dict[sig_results.tot_epochs]
    
    z_score = np.sqrt(2)*spc.erfinv((len(bkg_t[bkg_t<=np.median(sig_t)])/len(bkg_t))*2-1)
    return z_score
