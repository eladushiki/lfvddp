from pathlib import Path
import os,time
from mattiasdata import utils as u
import paper_scripts.train.save_jobs_script as jobs
from fractions import Fraction
import datetime

# todo: export all to configuration files
runtag="test"
train_sizes=['1/4'] #,'1/2']#['1/2']#[#['4/5','1/2','100/101','1/16','1/4']#['1/4','1/8','1/16','4/5','1/2','13/30']#['1/2','1/4','1/16','4/5','1/2','4/5','100/101']
test_sizes=['1/4'] #,'1/2']#['1/2']##['1/15', '1/24','1/5','1/101','1/16','1/4']#['1/4','1/8','1/16','1/15', '1/24','13/30']#['1/2','1/4','1/16','1/15', '1/24','1/5','1/101']
number_of_signals = [250,275]#[0,400,800,50,3000,2000,1600,1200,800,90,200,400]
walltime = ["12:00:00"] #,"30:00:00"]
save_walltime = "05:00"
remove = True
sample = "exp"
pdf = 0

def main():
    user = jobs.cmdline('whoami')[0][2:-3]
    try:
        if user=="yuvalzu":
            outdir = f"/srv01/agrp/yuvalzu/storage_links/NPLM_package"
            scriptsdir = "/srv01/agrp/yuvalzu/scripts/NPLM_package"
            configsdir = scriptsdir
        if user=="inbarsav":
            outdir = f"/srv01/tgrp/inbarsav/NPLM/NPLM_package"
            scriptsdir = "/srv01/tgrp/inbarsav/LFV_git/LFV_nn/LFV_nn"
            configsdir = scriptsdir
        else:  # Generic user, one step before moving this completely to a configuration file
            project_root = Path(__file__).parent.parent.parent.absolute()
            outdir = str(project_root / "results")
            scriptsdir = str(project_root / "paper_scripts/train")
            configsdir = str(project_root / "configs")
    except:
        print("invalid user")
    init_dir(outdir)
    submit_jobs(outdir,scriptsdir,configsdir)

def submit_jobs(outdir,scriptsdir,configsdir):
    for config_num in [22]:
        for sig in number_of_signals:
            N_jobs = 1 # todo: N_jobs = 500 if sig==0 else 100  # 500 for bg and 100 signal, to put in configurations file
            for i in range(len(train_sizes)):
                train_size=train_sizes[i]  # this is the proportional part from the data that is taken to be A/B out of some 200000 hardcoded number
                test_size=test_sizes[i]
                for jobid in range(N_jobs):
                    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
                    setupLines=[f"cd {outdir}"]
                    
                    # This is for sample A (bc includes signal)
                    fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_{jobid}_{i}_{sig}_{config_num}_TAU_{sample}_{pdf}.sh"
                    cmd_TAU=f"python {scriptsdir}/new_training.py -j {jobid} -R {train_size} -B {test_size} -s {sig} -c {configsdir}/config{config_num}.json -o {outdir}/training_outcomes --seed {seed} "
                    cmd_TAU+=f"-t TAU -S {sample} --BDstr Bkg --SDstr Sig --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
                    u.prepare_submit_file(fsubname,setupLines,[cmd_TAU],shortname=f"sub_{jobid}_{i}_{sig}_{config_num}_TAU_{sample}_{pdf}",setupATLAS=False)
                    u.submit_job(fsubname,walltime=walltime[i],mem=4)

                    # This is for sample B
                    fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_{jobid}_{i}_{sig}_{config_num}_delta_{sample}_{pdf}.sh"
                    cmd_delta=f"python {scriptsdir}/new_training.py -j {jobid} -R {train_size} -B {test_size} -s {sig} -c {configsdir}/config{config_num}.json -o {outdir}/training_outcomes --seed {seed} "
                    cmd_delta+=f"-t delta -S {sample} --BDstr Ref --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
                    u.prepare_submit_file(fsubname,setupLines,[cmd_delta],shortname=f"sub_{jobid}_{i}_{sig}_{config_num}_delta_{sample}_{pdf}",setupATLAS=False)
                    u.submit_job(fsubname,walltime=walltime[i],mem=4)

                fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_save_{i}_{sig}_{config_num}.sh"
                setupLines=[f"cd {outdir}"]
                cmd_to_save=f"python {scriptsdir}/new_make_jobs_tar+csv.py {train_size} {test_size} {sig} {N_jobs} {remove} {configsdir}/config{config_num}.json {outdir}/training_outcomes {sample} {pdf} {outdir}/training_outcomes"
                time.sleep(30)
                jobs.prepare_submit_files_save(fsubname,setupLines,[cmd_to_save],shortname=f"sub_save_{i}_{sig}_{config_num}",setupATLAS=False)
                N_jobs_wait = N_jobs+N_jobs
                jobs.submit_save_jobs(fsubname,N_jobs_wait,walltime=save_walltime,mem=2)

def init_dir(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if not os.path.isdir(outdir+'/'+runtag):
        os.makedirs(outdir+'/'+runtag)
    subdirs=['outputs','submit']
    for d in subdirs:
        if not os.path.isdir(outdir+'/'+runtag+'/'+d):
            os.makedirs(outdir+'/'+runtag+'/'+d)

if __name__=="__main__":
    main()
