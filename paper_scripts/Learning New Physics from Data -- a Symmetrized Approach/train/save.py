import sys,os,time
sys.path.insert(0,"/storage/agrp/yuvalzu/mattiasdata")
import frame.command_line.execution as u
import save_jobs_script as jobs
from fractions import Fraction
from datetime import datetime
from subprocess import run
from subprocess import PIPE, Popen
import json

remove = False
train_sizes=['1/4']#['1/2','4/5','4/5','1/2']#['1/2','1/4','1/16','4/5','4/5','1/2']
test_sizes=['1/4']#['1/2','1/5','1/15', '1/24']#['1/2','1/4','1/16','1/5','1/15', '1/24']
outdir = f"/srv01/tgrp/inbarsav/NPLM/NPLM_package"
scriptsdir = "/srv01/tgrp/inbarsav/LFV_git/LFV_nn/LFV_nn"
N_jobs = 2000
DIR = "/srv01/tgrp/inbarsav/NPLM/NPLM_package/training_outcomes/"
sample = "exp"
pdf = 0

#with open('/srv01/agrp/yuvalzu/scripts/NPLM_package/initial_config.json', 'r') as initialjsonfile:
    #initial_config = json.load(initialjsonfile)

with open('/srv01/tgrp/inbarsav/LFV_git/LFV_nn/LFV_nn/initial_config.json', 'r') as initialjsonfile:
    initial_config = json.load(initialjsonfile)    

for config_numb in ["signals_1.6Sig_loc_0.16Sig_scale_Trueresonant"]:#["_NPLM_wc_9"]:
    for sig in [300,0]:
        N_jobs = 2000 if sig==300 else 2000
        config_num = config_numb
        config_num = "signals" if sig==0 else config_numb
        config_num = f"_{sig}"+config_num
# for config_num in [21]:
#     jsonfile = f"{scriptsdir}/config{config_num}.json"
#     with open(jsonfile, 'r') as jsonfile:
#         config = json.load(jsonfile)
#     for sig in [300]:#[20,30]:#:#:number_of_signals: [3000,2000,1600,1200,800,90,200,400]
        for i in range(len(train_sizes)):
            train_size=train_sizes[i]
            test_size=test_sizes[i]
            try:
                #OUTPUT_FILE = DIR+'exp'+''.join(list((f"{config[key]}{key}" if config[key]!=initial_config[key] else '' for key in config.keys())))+f'{sig}signals_{train_size}Ref_{test_size}Bkg'.replace('/',':')
                #os.system(f"tar -cz --force-local -f {OUTPUT_FILE}.tar.gz {OUTPUT_FILE}*.h5")
                #os.system(f"/usr/local/anaconda/3.8/bin/python /srv01/agrp/yuvalzu/scripts/terminal_scripts/make_jobs_tar+csv_copy.py {train_size} {test_size} {sig} {N_jobs} {remove} {scriptsdir}/config{config_num}.json {outdir}/training_outcomes")
                #os.system(f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_make_jobs_tar+csv.py {train_size} {test_size} {sig} {N_jobs} {remove} {scriptsdir}/config{config_num}.json {outdir}/training_outcomes {sample} {pdf} {scriptsdir}/training_outcomes")
                #os.system(f"/usr/local/anaconda/3.8/bin/python /srv01/agrp/yuvalzu/scripts/terminal_scripts/new_make_jobs_tar+csv.py {train_size} {test_size} {sig} {N_jobs} {remove} {scriptsdir}/config{config_num}.json {outdir}/training_outcomes {sample} {pdf}")
                os.system(f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_make_jobs_tar+csv.py {train_size} {test_size} {sig} {N_jobs} {remove} {scriptsdir}/config{config_num}.json {outdir}/training_outcomes {sample} {pdf} {scriptsdir}/training_outcomes")


                #cmd_to_save=f"/usr/local/anaconda/3.8/bin/python /srv01/agrp/yuvalzu/scripts/terminal_scripts/make_jobs_tar+csv_copy.py {train_size} {test_size} {sig} {N_jobs} {remove} {scriptsdir}/config{config_num}.json {outdir}/training_outcomes"
            except:
                print("missing:"+f"{train_size} config:{config_num}")