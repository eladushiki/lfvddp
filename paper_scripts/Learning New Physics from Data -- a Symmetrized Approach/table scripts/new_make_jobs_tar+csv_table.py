### last line of the script removes the .h5 and .txt files!!!
import os,sys,json,argparse,glob
import pandas as pd
from subprocess import PIPE, Popen
import tarfile

train_size = sys.argv[1]
test_size = sys.argv[2]
signals = sys.argv[3]
N_jobs = sys.argv[4]
remove = sys.argv[5]=='True'
jsonfile = sys.argv[6]
outdir = sys.argv[7]
pdf = int(sys.argv[8])
scriptsdir = sys.argv[9]
# remove  = False 

# parser = argparse.ArgumentParser()    
# parser.add_argument('-j', '--jsonfile'  , type=str, help="json file", required=True)
# args = parser.parse_args()

## Import parameters from the give json file
with open(jsonfile, 'r') as jsonfile:
    config = json.load(jsonfile)

initial_jsonfile = '/srv01/agrp/yuvalzu/scripts/NPLM_package/new_initial_config.json' if 'yuvalzu' in outdir else '/srv01/tgrp/inbarsav/LFV_git/LFV_nn/LFV_nn/new_initial_config.json'
with open(initial_jsonfile, 'r') as js:
    initial_config = json.load(js)

def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        stderr=PIPE,
    shell=True)
    out,err=process.communicate()
    returncode=process.returncode
    return str(out),str(err),returncode

OUTPUT_PATH = outdir
# OUTPUT_FILE = f'em500000epochs{signals}signals_{train_size}Ref_{test_size}Bkg'.replace('/',':')
OUTPUT_FILE = ''.join(list((f"{key}-{config[key]}_" if config[key]!=initial_config[key] else '' for key in config.keys())))+f'{signals}signals_{train_size}Ref_{test_size}Bkg'.replace('/',':')

os.chdir(OUTPUT_PATH)
os.system(f"/usr/local/anaconda/3.8/bin/python /srv01/agrp/yuvalzu/scripts/terminal_scripts/copy_txt_to_csv.py {OUTPUT_FILE} {OUTPUT_FILE}")
os.system(f"cp {outdir}/{OUTPUT_FILE}.csv {scriptsdir}")
os.system(f"tar -cz --force-local -f {OUTPUT_FILE}.tar.gz {OUTPUT_FILE}*.h5")
### next line removes the .h5 and .txt files!!!
if remove: 
    os.system(f"rm -r {OUTPUT_FILE}*.txt {OUTPUT_FILE}*.h5")

# read logs and save relevant ones
log_dir = OUTPUT_PATH.replace("training_outcomes","test/submit")
log_files = glob.glob(f"{log_dir}/*.log")
if os.path.exists(log_dir.replace("test","test_old")):
    log_files = log_files + glob.glob(f'{log_dir.replace("test","test_old")}/*.log')
if len(log_files)>0:
    log_files_to_save=[]
    for log_file in log_files:
        with open(log_file,'r') as f:
            contents = f.read()
            if OUTPUT_FILE in contents:
                log_files_to_save.append(log_file)
    if len(log_files_to_save)>0:
        with tarfile.open(f'{OUTPUT_FILE}_logs.tar.gz', "w:gz") as tar:
            for file in log_files_to_save:
                tar.add(file, os.path.basename(file))
                if remove: 
                    os.system(f"rm -r {file}")
                    if os.path.exists(file.replace("log","sh")):
                        os.system(f'rm -r {file.replace("log","sh")}')

# add config information to the results csv file
shared_results = pd.read_csv(f'{scriptsdir}/results.csv')
personal_results = pd.read_csv(f'{outdir}/results.csv')
config_df = pd.DataFrame([config])
config_df['A_size'] = train_size
config_df['B_size'] = test_size
config_df['sig_events'] = signals
config_df['csv_file'] = f'{OUTPUT_FILE}.csv'
config_df['tar_file'] = f'{OUTPUT_FILE}.tar.gz'
shared_results = pd.concat([shared_results,config_df],ignore_index=True)
shared_results.to_csv(f'{scriptsdir}/results.csv',index=False)
personal_results = pd.concat([personal_results,config_df],ignore_index=True)
personal_results.to_csv(f'{outdir}/results.csv',index=False)



