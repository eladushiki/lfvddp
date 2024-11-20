import sys,os,time
sys.path.insert(0,"/storage/agrp/yuvalzu/mattiasdata")
import frame.submit
import save_jobs_script as jobs
from fractions import Fraction
import datetime
from subprocess import run
from subprocess import PIPE, Popen

runtag="test"
train_sizes=['5/8','3/4']#['1/2']#[#['4/5','1/2','100/101','1/16','1/4']#['1/4','1/8','1/16','4/5','1/2','13/30']#['1/2','1/4','1/16','4/5','1/2','4/5','100/101']
test_sizes=['5/8','3/4']#['1/2']##['1/15', '1/24','1/5','1/101','1/16','1/4']#['1/4','1/8','1/16','1/15', '1/24','13/30']#['1/2','1/4','1/16','1/15', '1/24','1/5','1/101']
BR = [1000/456,1600/456,2800/456]
#number_of_signals = [1000,1200]
walltime = ["50:00:00","50:00:00"]
save_walltime = "01:00:00"
remove = True
sample = "em_Mcoll"
pdf = 0

def main():
    user = jobs.cmdline('whoami')[0][2:-3]
    try:
        if user=="yuvalzu":
            outdir = f"/srv01/agrp/yuvalzu/storage_links/NPLM_package"
            scriptsdir = "/srv01/agrp/yuvalzu/scripts/NPLM_package"
        if user=="inbarsav":
            outdir = f"/srv01/tgrp/inbarsav/NPLM/NPLM_package"
            scriptsdir = "/srv01/tgrp/inbarsav/LFV_git/LFV_nn/LFV_nn"
    except:
        print("invalid user")
    init_dir(outdir)
    submit_jobs(outdir,scriptsdir)

def submit_jobs(outdir,scriptsdir):
    for config_num in ['_Mcoll']:
        for i in range(len(train_sizes)):
            train_size=train_sizes[i]
            test_size=test_sizes[i]
            for br in BR:
                sig = round(float(br)*Fraction(test_size)*456)
                N_jobs = 200 if sig==0 else 200
                for jobid in range(N_jobs):
                    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
                    setupLines=[f"cd {outdir}"]
                    fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_{jobid}_{i}_{sig}_{config_num}_TAU_{sample}_{pdf}.sh"
                    cmd_TAU=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_training.py -j {jobid} -R {train_size} -B {test_size} -s {sig} -c {scriptsdir}/config{config_num}.json -o {outdir}/training_outcomes --seed {seed} "
                    cmd_TAU+=f"-t TAU -S {sample} --BDstr Bkg --SDstr Sig --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
                    frame.submit.prepare_submit_file(fsubname,setupLines,[cmd_TAU],shortname=f"sub_{jobid}_{i}_{sig}_{config_num}_TAU_{sample}_{pdf}",setupATLAS=False)
                    frame.submit.submit_job(fsubname,walltime=walltime[i],io=0,mem=4)

                    fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_{jobid}_{i}_{sig}_{config_num}_delta_{sample}_{pdf}.sh"
                    cmd_delta=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_training.py -j {jobid} -R {train_size} -B {test_size} -s {sig} -c {scriptsdir}/config{config_num}.json -o {outdir}/training_outcomes --seed {seed} "
                    cmd_delta+=f"-t delta -S {sample} --BDstr Ref --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
                    frame.submit.prepare_submit_file(fsubname,setupLines,[cmd_delta],shortname=f"sub_{jobid}_{i}_{sig}_{config_num}_delta_{sample}_{pdf}",setupATLAS=False)
                    frame.submit.submit_job(fsubname,walltime=walltime[i],io=0,mem=4)

                fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_save_{i}_{sig}_{config_num}.sh"
                setupLines=[f"cd {outdir}"]
                cmd_to_save=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_make_jobs_tar+csv.py {train_size} {test_size} {sig} {N_jobs} {remove} {scriptsdir}/config{config_num}.json {outdir}/training_outcomes {sample} {pdf} {scriptsdir}/training_outcomes"
                time.sleep(30)
                jobs.prepare_submit_files_save(fsubname,setupLines,[cmd_to_save],shortname=f"sub_save_{i}_{sig}_{config_num}",setupATLAS=False)
                N_jobs_wait = N_jobs+N_jobs
                jobs.submit_save_jobs(fsubname,N_jobs_wait,walltime=save_walltime,io=0,mem=2)

def init_dir(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    subdirs=['outputs','submit']
    for d in subdirs:
        if not os.path.isdir(outdir+'/'+d):
            os.makedirs(outdir+'/'+d)

if __name__=="__main__":
    main()
