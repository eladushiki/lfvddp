import numpy as np, os, re, glob, h5py, sys, os, time, tarfile, pandas as pd
from scipy.stats import chi2
import scipy.special as spc
from scipy.integrate import quad

sys.path.insert(0,"/storage/agrp/yuvalzu/mattiasdata")
import save_jobs_script as jobs
import frame.command_line.execution as u

#------------------------------------------------------------------------------------------------------------------------#

def user_dir():
    '''
    returns the training results directory of the user
    '''
    global dirYuval, dirInbar
    dirYuval = "/srv01/agrp/yuvalzu/storage_links/NPLM_package/training_outcomes/"
    dirInbar = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
    user = jobs.cmdline('whoami')[0][2:-3]
    try:
        if user=="yuvalzu":
            dir = dirYuval
        if user=="inbarsav":
            dir = dirInbar
    except:
        print("invalid user")
    return(dir)

def user_plots_dir():
    '''
    returns the plots directory of the user
    '''
    plots_dirYuval = '/srv01/agrp/yuvalzu/scripts/NPLM_package/plots/'
    plots_dirInbar = '/srv01/tgrp/inbarsav/LFV_git/LFV_nn/LFV_nn/plots/' 
    user = jobs.cmdline('whoami')[0][2:-3]
    try:
        if user=="yuvalzu":
            plots_dir =  plots_dirYuval
        if user=="inbarsav":
            plots_dir =  plots_dirInbar
    except:
        print("invalid user")
    return(plots_dir)


def t_hist_epoch(t_history,epochs_list,epoch_numbers):
    '''
    returns a dictionary of the t values for each epoch in epoch_numbers.
    
    Parameters
    ----------
    t_history : list of numpy arrays
        list of the t values of each epoch (each value comes from diffrent h5 file in the tar.gz and of different epoch).
    epochs_list : list of numpy arrays
        list of the epochs corresponding to each t value in t_history.
    epoch_numbers : list 
        list of the epochs to return their t values.
    
    Returns
    -------
    t_hist_dict : dictionary
        dictionary of the t values for each epoch in epoch_numbers.
    '''
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

def get_results_table():
    '''
    returns a pandas dataframe of the results.csv file.
    '''
    results_table_yuval = pd.read_csv(dirYuval+"results.csv", keep_default_na=False)
    # results_table_inbar = pd.read_csv(dirInbar+"results.csv")
    # results_table = pd.concat([results_yuval,results_inbar],ignore_index=True)
    # return results_table
    return results_table_yuval

class results:
    N = 219087    # number of background events (219087 for em channel, used also for exponential distribution)
    dir = user_dir()
    def __init__(self,file_name):  
        self.file = file_name
        self.tar_file_name = file_name.replace(".csv",".tar.gz") if file_name.endswith(".csv") else file_name
        self.csv_file_name = file_name.replace(".tar.gz",".csv") if file_name.endswith(".tar.gz") else file_name
        self.results_table = get_results_table()
        self.file_settings = self.results_table[self.results_table['csv_file']==self.csv_file_name]    # get the row in the results table that corresponds to the file
        total_factor = self.file_settings['CP'].values[0]
        self.total_events = total_factor*results.N
        ref_str = self.file_settings['A_size'].values[0].split('/')
        self.Ref_ratio = total_factor*float(ref_str[0])/float(ref_str[1]) if len(ref_str)>1 else total_factor*float(ref_str[0])
        self.Ref_events = int(results.N*self.Ref_ratio)
        bkg_str = self.file_settings['B_size'].values[0].split('/')
        self.Bkg_ratio = total_factor*float(bkg_str[0])/float(bkg_str[1]) if len(bkg_str)>1 else total_factor*float(bkg_str[0])
        self.Bkg_events = int(results.N*self.Bkg_ratio)
        self.Sig_events = int(self.file_settings['sig_events'].values[0])
        self.Bkg_sample = self.file_settings['ch'].values[0]
        self.binned = "True" if self.file_settings['binned'].values[0]=='True' else "False"    # True if we limited the training data resolution
        self.resolution = float(self.file_settings['resolution'].values[0]) if float(self.file_settings['resolution'].values[0])!=0.0 else 0    # the resolution of the data for binning
        self.WC = self.file_settings['WC'].values[0] if self.file_settings['WC'].values[0]!='None' else float(self.file_settings['WC'].values[0])    # range for NN's weights, "None" if unlimited.
        self.N_poiss = "True" if self.file_settings['poiss'].values[0]=='True' else "False"    # if True, the number of events in Ref, Bkg and Sig is poisson distributed around Ref_events, Bkg_events and Sig_events, respectively.
        self.NPLM = "True" if self.file_settings['NPLM'].values[0]==True else "False"    # if True, we used the NPLM definition for t (the test statistic).
        self.Sig_resonant = "True" if self.file_settings['resonant'].values[0]=='True' else "False"    # for exponential distribution: if True, the signal is a gaussian, otherwise it is non-resonant. 
        self.Sig_loc = float(self.file_settings['Sig_loc'].values[0])    # location of the gaussian signal (for exponential distribution)
        self.Sig_scale = float(self.file_settings['Sig_scale'].values[0])    # scale of the gaussian signal (for exponential distribution)
        self.resample = "True" if self.file_settings['resample'].values[0]=='True' else "False"   # see docstring of resample function in new_setting.py
        if self.resample=="True":
            self.label_method = self.file_settings['label_method'].values[0]
            self.N_method = self.file_settings['N_method'].values[0]
            self.replacement = "True" if self.file_settings['replacement'].values[0]=='True' else "False"
            self.original_seed = int(self.file_settings['original_seed'].values[0])
        else:
            self.label_method = ""
            self.N_method = ""
            self.replacement = ""
            self.original_seed = None

        self.tot_epochs = int(self.file_settings['epochs'].values[0])
        self.patience = int(self.file_settings['patience'].values[0])


    def get_similar_files(self,epochs='all',patience='all'):
        '''
        get all files with the same parameters as self, but are with different names or in different directories.
        
        Parameters
        ----------
        epochs : int or 'all'
            if int, get only files with the same number of epochs as "epochs" in their names (epochs_tau).
        patience_tau : int or 'all'
            if int, get only files with the same patience_tau as "patience_tau" in their names.
        
        Returns
        -------
        similar_files : list
            list of all files with the same parameters as self.
        '''
        global parameters_list
        parameters_list = self.results_table.columns.values.tolist()
        ignore_list = ['run','csv_file','tar_file']    # parameters to ignore in the search for similar files
        if epochs=='all': ignore_list.append('epochs')
        if patience=='all': ignore_list.append('patience')
        parameters_list = [parameter for parameter in parameters_list if parameter not in ignore_list]
        similar_files = self.results_table[self.results_table[parameters_list].eq(self.file_settings[parameters_list].values[0]).all(axis=1)]['csv_file'].values.tolist()
        similar_files = [user_dir()+file.rsplit('.',1)[0] for file in similar_files]
        return similar_files
    
    
    def __len__(self):
        '''
        returns the number of toys used in self and its similar files.
        '''
        count = 0
        for file in self.get_similar_files(epochs=self.tot_epochs):
            file = file.replace(".csv",".tar.gz") if ".csv" in file else file if ".tar.gz" in file else file+".tar.gz"
            with tarfile.open(file,"r:gz") as tar:
                count += len(tar.getnames())
        return count//4 if self.NPLM=="False" else count//2   # 4 for TAU/delta and history/weights options of the h5.

    
    def read_final_t_csv(self):
        '''
        read the final t values from the csv file of self.

        Returns
        -------
        TAU_plus_delta : numpy array
            array of the final t values (t=TAU+delta).
        delta_names : list
            list of the names of the txt files that was saved in the csv file, corresponding to their final t values.
        '''
        TAU_names = []
        TAUs = []
        delta_names = []
        deltas = []
        TAU_plus_delta =[]
        for file in self.get_similar_files(epochs=self.tot_epochs):
            if ".csv" not in file:
                file = file.replace("tar.gz","csv") if ".tar.gz" in file else file+".csv"
            csv_file = file
            file = csv_file.split(".csv")[0].split("/")[-1]
            if os.path.exists(csv_file):
                with open(csv_file,'r') as f:
                    lines = f.readlines()
                    TAU_names += [tau.split(',')[1] for tau in lines if ('TAU.' in tau and file in tau)]
                    TAUs += [float(tau.split(',')[0]) for tau in lines if ('TAU.' in tau and file in tau)]
                    # TAUs = np.array([float(tau.split(',')[0]) for tau in lines if tau.count('TAU.')])
                    delta_names += [delta.split(',')[1] for delta in lines if ('delta.' in delta and file in delta)]
                    deltas += [float(delta.split(',')[0]) for delta in lines if ('delta.' in delta and file in delta)]
                    # deltas = np.array([float(delta.split(',')[0]) for delta in lines if delta.count('delta.')])
                    
        if self.NPLM=="True":
            TAU_plus_delta =np.array(TAUs)
            delta_names =TAU_names.copy()
        else:            
            TAU_plus_delta = np.array([TAUs[TAU_names.index(delta_name.replace("delta.txt","TAU.txt"))]+deltas[delta_names.index(delta_name)] for delta_name in delta_names if delta_name.replace("delta.txt","TAU.txt") in TAU_names])
            TAU_plus_delta_names = [TAU_names[TAU_names.index(delta_name.replace("delta.txt","TAU.txt"))]+' + '+delta_names[delta_names.index(delta_name)] for delta_name in delta_names if delta_name.replace("delta.txt","TAU.txt") in TAU_names]
            delta_names = [delta_names[delta_names.index(delta_name)] for delta_name in delta_names if delta_name.replace("delta.txt","TAU.txt") in TAU_names]

        return TAU_plus_delta, delta_names
            
    
    def get_t_history(self):
        '''
        get the t values from the tar file of self.

        Returns
        -------
        t_history : list of numpy arrays
            list of the t values of each epoch (each value comes from diffrent h5 file in the tar.gz and of different epoch).
        epochs : list of numpy arrays
            list of the epochs of each t value in t_history.
        seeds : list of numpy arrays
            list of the seeds of each h5 file in the tar.gz.
        '''
        dir = results.dir
        similar_files= self.get_similar_files(epochs=self.tot_epochs)
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
                # patience_tau = float((re.search(r'\d+patience_tau',filename).group()).split("patience")[0] if "patience_tau" in filename else 1000) 
                # patience_delta = float((re.search(r'\d+patience_delta',filename).group()).split("patience")[0] if "patience_delta" in filename else 1000)
                # patience = max(patience_tau,patience_delta)
                patience = self.patience
                # step_tau = round(patience/patience_tau)
                # step_delta = round(patience/patience_delta)
                step = 1
                if self.NPLM=="False":
                    if (('_TAU_history' in filename) and (filename.replace('_TAU_history','_delta_history') in files)):
                        with h5py.File(filename, "r") as f1:
                            keys_list  = [(key) for key in list(f1.keys())]
                            TAU_history = f1.get(str(keys_list[2])) #'loss'
                            TAU_history = np.array(TAU_history)
                        with h5py.File(filename.replace('_TAU_history','_delta_history'), "r") as f2:
                            keys_list  = [(key) for key in list(f2.keys())]
                            delta_history = f2.get(str(keys_list[2])) #'loss'
                            delta_history = np.array(delta_history)
                        seed_num = int(re.search(r'_seed\d+_',filename)[0][len('_seed'):-len('_')])
                        # t_history.append(-2*(TAU_history[0::step_tau]+delta_history[0::step_delta]))
                        t_history.append(-2*(TAU_history[0::step]+delta_history[0::step]))
                        # epochs.append(patience*np.array(range(len(TAU_history[0::step_tau]))))
                        epochs.append(patience*np.array(range(len(TAU_history[0::step]))))
                        # seeds.append(seed_num*np.ones_like(np.array(range(len(TAU_history[0::step_tau])))))
                        seeds.append(seed_num*np.ones_like(np.array(range(len(TAU_history[0::step])))))
                elif self.NPLM=="True":
                    if '_TAU_history' in filename:
                        with h5py.File(filename, "r") as f1:
                            keys_list  = [(key) for key in list(f1.keys())]
                            TAU_history = f1.get(str(keys_list[2])) #'loss'
                            TAU_history = np.array(TAU_history)
                        seed_num = int(re.search(r'_seed\d+_',filename)[0][len('_seed'):-len('_')])
                        # t_history.append(-2*(TAU_history[0::step_tau]))
                        t_history.append(-2*(TAU_history[0::step]))
                        # epochs.append(patience*np.array(range(len(TAU_history[0::step_tau]))))
                        epochs.append(patience*np.array(range(len(TAU_history[0::step]))))
                        # seeds.append(seed_num*np.ones_like(np.array(range(len(TAU_history[0::step_tau])))))
                        seeds.append(seed_num*np.ones_like(np.array(range(len(TAU_history[0::step])))))
        #os.system(f'rm {dir}extract_here/*')
        if len(t_final)>0:
            for i,t in enumerate(t_final):
                    name = txt_names[i]
                    seed_num = int(re.search(r'_seed\d+_',name)[0][len('_seed'):-len('_')])
                    # tot_epochs = float(re.search(r'\d+epochs_tau',name)[0][:-len('epochs_tau')])
                    tot_epochs = self.tot_epochs
                    t_history.append(np.array([t]))
                    epochs.append(np.array([tot_epochs]))
                    seeds.append(seed_num*np.ones_like(np.array([tot_epochs])))
        os.system(f'rm -r {dir}extract_here')
        os.system(f"mkdir {dir}extract_here")
        return t_history,epochs,seeds
    
    def get_t_history_dict(self):
        '''
        get a dictionary of epoch-t value pairs for every saved epoch in the training.

        Returns
        -------
        t_hist_dict : dictionary
        '''
        t_sig_hist, epochs_sig_list, seeds_list = self.get_t_history()
        epochs_list = np.unique(np.concatenate(epochs_sig_list).ravel())
        Sig_t = t_hist_epoch(t_sig_hist, epochs_sig_list, epochs_list)
        self.t_history = Sig_t.copy()
        return Sig_t
    
    def get_signal_files(self,N_sig='all',Sig_loc='all',Sig_scale='all',resonant='all'):
        '''
        get all signal files with the same parameters as self, the background file.

        Parameters
        ----------
        N_sig : int or 'all'
            if int, get only files with the same number of signal events as N_sig in their names.
        Sig_loc : float or 'all'
            for exponential distribution: if float, get only files with the same signal location as Sig_loc in their names.
        Sig_scale : float or 'all'
            for exponential distribution: if float, get only files with the same signal scale as Sig_scale in their names.
        resonant : str
            for exponential distribution: if 'True', get only files with resonant signal. if 'False', get only files with non-resonant signal. if 'all', get both.

        Returns
        -------
        filenames : list
            list of all signal files with the same parameters as self.
        '''
        filenames = []
        similar_filenames = self.get_similar_files(epochs=self.tot_epochs)
        bkg_search_filename = self.similar_search_name.replace("tar.gz","csv")
        if self.WC!=9:
            bkg_search_filename =(bkg_search_filename.split('clipping')[0]+'*'+bkg_search_filename.split('signals_')[1])
            sig_filename = re.sub('\*\*+', '*', bkg_search_filename)
        else:
            sig_filename = '*'+bkg_search_filename.split('signals_')[1]

        sig_files = glob.glob(results.dirYuval+sig_filename)+glob.glob(results.dirInbar+sig_filename)
        for file in sig_files:
            file = file.split('/')[-1]
            params_file = results(file)
            flag = False
            if (params_file.Bkg_events==self.Bkg_events) and (params_file.Ref_events==self.Ref_events) and (params_file.N_poiss==self.N_poiss) and (params_file.resolution==self.resolution) and (params_file.NPLM==self.NPLM) and (params_file.Bkg_sample==self.Bkg_sample) and (params_file.WC==self.WC):
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

    def remove(self):
        '''
        remove the results files of self, their logs, and delete the corresponding row in the results.csv table.
        '''
        confirm = input(f"Are you sure you want to remove the files of {self.csv_file_name[:-4]}? (y/n)")
        while confirm not in ['y','n']:
            print("invalid input, please enter 'y' or 'n'")
            confirm = input(f"Are you sure you want to remove the files of {self.csv_file_name[:-4]}? (y/n)")
        if confirm == 'n':
            return
        if confirm == 'y':
            os.system(f'rm {results.dir+self.csv_file_name[:-4]}*')
            os.system(f'rm {user_plots_dir()[:-6]+"training_outcomes/"+self.csv_file_name[:-4]}*')
            personal_results_table = self.results_table[self.results_table['csv_file']!=self.csv_file_name]
            personal_results_table.to_csv(results.dir+'results.csv',index=False)
            joint_results_table = pd.read_csv(f'{user_plots_dir()[:-6]+"training_outcomes/"}'+'results.csv')
            joint_results_table = joint_results_table[joint_results_table['csv_file']!=self.csv_file_name]
            joint_results_table.to_csv(f'{user_plots_dir()[:-6]+"training_outcomes/"}'+'results.csv',index=False)
            print('removed files')



class exp_results(results):
    def __init__(self, file_name):
        super().__init__(file_name)
        # self.Sig_resonant = "False" if "Falseresonant" in file_name else "True"
        # self.Sig_loc = float(re.search(r'\d+.?\d*Sig_loc', file_name)[0][:-len('Sig_loc')]) if 'Sig_loc' in file_name else 6.4
        # self.Sig_scale = float(re.search(r'\d+.?\d*Sig_scale', file_name)[0][:-len('Sig_scale')]) if 'Sig_scale' in file_name else 0.16

    def get_sqrt_q0(self):
        '''
        returns the square root of the q0 (optimal) test for the exponential distribution, for both resonant and non-resonant signals.
        '''
        Bkg_pdf = lambda x: np.exp(-x)
        if self.Sig_resonant == "False":
            Sig_pdf = lambda x: x**2*np.exp(-x)/2
            integrand = lambda x: (self.Sig_events*Sig_pdf(x)+self.Bkg_events*Bkg_pdf(x))*np.log(1+self.Sig_events*Sig_pdf(x)/(self.Bkg_events*Bkg_pdf(x)))
            sqrt_q0 = -2*(self.Sig_events-(quad(integrand, 0, 200)[0]))    # 200 is the upper limit of the integral to avoid numpy's infinities.
        else:
            Sig_loc = self.Sig_loc
            sigma = self.Sig_scale
            Sig_pdf = lambda x: (1/(np.sqrt(2*np.pi)*sigma))*np.exp(-(x-Sig_loc)**2/(2*sigma**2))
            integrand = lambda x: (self.Sig_events*Sig_pdf(x)+self.Bkg_events*Bkg_pdf(x))*np.log(1+self.Sig_events*Sig_pdf(x)/(self.Bkg_events*Bkg_pdf(x)))
            sqrt_q0 = -2*(self.Sig_events-(quad(integrand, 0, Sig_loc)[0]+quad(integrand, Sig_loc, np.inf)[0]))
        return np.sqrt(sqrt_q0)

    def get_signals_files(self):
        '''
        returns lists of all signals files, for the 3 signals used in the paper.
        '''
        filenames1 =self.get_signal_files(Sig_loc = [6.4],Sig_scale = [0.16],resonant=["True"])
        filenames2 =self.get_signal_files(resonant=["False"])
        filenames3 =self.get_signal_files(Sig_loc = [1.6],Sig_scale = [0.16],resonant=["True"])
        return filenames1, filenames2, filenames3



class em_results(results):
    def __init__(self, file_name):
        super().__init__(file_name)

    def collect_data(self):
        '''
        returns the background and signal numpy arrays.
        should be modified to work for all channels, all signals and all variables.
        '''
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

    def get_sqrt_q0(self,Data_bins=6):
        Bkg, signal, signal_samples = self.collect_data()
        Sig = np.concatenate(tuple([signal[f"{s}_em_signal"] for s in signal_samples]),axis=0).reshape(-1,)
        mu = self.Sig_events/len(Sig)
        bkgFrac = self.Bkg_events/len(Bkg)
        histData = np.histogram(Sig,Data_bins)
        bins = histData[1]
        data = histData[0][histData[0]!=0]
        histBkg = np.histogram(Bkg,bins)
        bkg = histBkg[0][histData[0]!=0]
        sqrt_q0 = 2*(-self.Sig_events+np.sum((mu*data+(bkg*bkgFrac))*np.log(mu*data/(bkg*bkgFrac)+1)))
        return np.sqrt(sqrt_q0)
    
    def get_binned_sqrt_q0(self,resolution=0.05):
        Bkg, signal, signal_samples = self.collect_data()
        Sig = np.concatenate(tuple([signal[f"{s}_em_signal"] for s in signal_samples]),axis=0).reshape(-1,)
        Bkg = np.floor((Bkg/1e5)/resolution)*resolution
        Sig = np.floor((Sig/1e5)/resolution)*resolution
        mu = self.Sig_events/len(Sig)
        bkgFrac = self.Bkg_events/len(Bkg)
        decimals = len(str(resolution).split('.')[1])
        histData = np.histogram(Sig, bins=[min(Sig)+round(i*resolution,decimals) for i in range(int((max(Sig)-min(Sig)+round(2*resolution,decimals))/resolution))])
        bins = histData[1]
        data = histData[0][histData[0]!=0]
        histBkg = np.histogram(Bkg,bins)
        bkg = histBkg[0][histData[0]!=0]
        sqrt_q0 = 2*(-self.Sig_events+np.sum((mu*data+(bkg*bkgFrac))*np.log(mu*data/(bkg*bkgFrac)+1)))
        return np.sqrt(sqrt_q0)
    
    def get_signals_files(self):
        filenames =self.get_signal_files(Sig_loc = [6.4],Sig_scale = [0.16],resonant=["True"])
        return filenames
    


def get_z_score(results_file:results,bkg_results_file:results, epoch = 500000):
    sig_results = results_file
    bkg_results = bkg_results_file
    sig_file_name = sig_results.csv_file_name
    bkg_file_name = bkg_results_file.csv_file_name
    sig_t = []
    bkg_t = []
    z_score = float("nan")
    if epoch == sig_results.tot_epochs:
        sig_t = sig_results.read_final_t_csv()[0]
    if (len(sig_t)<1) or (epoch < sig_results.tot_epochs):
        sig_t_dict = sig_results.get_t_history_dict()
        sig_t = sig_t_dict[epoch] if epoch in sig_t_dict.keys() else sig_t_dict[sig_results.tot_epochs]
    if epoch > sig_results.tot_epochs: 
        print(f"maximal number of signal epochs = {sig_results.tot_epochs}")
        return z_score, sig_t, bkg_t

    if epoch == bkg_results.tot_epochs:
        bkg_t = bkg_results.read_final_t_csv()[0]
    if (len(bkg_t)<1) or (epoch < bkg_results.tot_epochs):
        bkg_t_dict = bkg_results.get_t_history_dict()
        bkg_t = bkg_t_dict[epoch] if epoch in bkg_t_dict.keys() else bkg_t_dict[sig_results.tot_epochs]
    if epoch > bkg_results.tot_epochs: 
        print(f"maximal number of bkg epochs = { bkg_results_file.tot_epochs}")
        return z_score, sig_t, bkg_t

    # replace NaNs in bkg_t and sig_t with inf.
    bkg_t[np.isnan(bkg_t)] = np.inf
    sig_t[np.isnan(sig_t)] = np.inf
    z_score = np.sqrt(2)*spc.erfinv((len(bkg_t[bkg_t<=np.median(sig_t)])/len(bkg_t))*2-1)
    return z_score, sig_t, bkg_t
