import sys,os,time
sys.path.insert(0,"/storage/agrp/yuvalzu/mattiasdata")
import utils as u
import save_jobs_script as jobs
from fractions import Fraction
import datetime
from subprocess import run
from subprocess import PIPE, Popen

runtag="test"
A_sizes=['1/2']#['1/2']#[#['4/5','1/2','100/101','1/16','1/4']#['1/4','1/8','1/16','4/5','1/2','13/30']#['1/2','1/4','1/16','4/5','1/2','4/5','100/101']
B_sizes=['1/2']#['1/2']##['1/15', '1/24','1/5','1/101','1/16','1/4']#['1/4','1/8','1/16','1/15', '1/24','13/30']#['1/2','1/4','1/16','1/15', '1/24','1/5','1/101']
number_of_signals = [0]#[0,400,800,50,3000,2000,1600,1200,800,90,200,400]
walltime = ["2:00:00"]
save_walltime = "01:00:00"
remove = True
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
    for config_num in ['_phi']:
        for sig in number_of_signals:
            N_jobs = 250 if sig==0 else 100
            for i in range(len(A_sizes)):
                A_size=A_sizes[i]
                B_size=B_sizes[i]
                for jobid in range(N_jobs):
                    seed = datetime.datetime.now().microsecond+datetime.datetime.now().second+datetime.datetime.now().minute
                    setupLines=[f"cd {outdir}"]
                    fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_{jobid}_{i}_{sig}_{config_num}_TAU_{pdf}.sh"
                    cmd_TAU=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_training_table.py -j {jobid} -A {A_size} -B {B_size} -s {sig} -c {scriptsdir}/config{config_num}.json -o {outdir}/training_outcomes --seed {seed} "
                    cmd_TAU+=f"-t TAU --BDstr Bkg --SDstr Sig --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
                    u.prepare_submit_file(fsubname,setupLines,[cmd_TAU],shortname=f"sub_{jobid}_{i}_{sig}_{config_num}_TAU_{pdf}",setupATLAS=False)
                    u.submit_job(fsubname,walltime=walltime[i],io=0,mem=2)

                    fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_{jobid}_{i}_{sig}_{config_num}_delta_{pdf}.sh"
                    cmd_delta=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_training_table.py -j {jobid} -A {A_size} -B {B_size} -s {sig} -c {scriptsdir}/config{config_num}.json -o {outdir}/training_outcomes --seed {seed} "
                    cmd_delta+=f"-t delta --BDstr Ref --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
                    u.prepare_submit_file(fsubname,setupLines,[cmd_delta],shortname=f"sub_{jobid}_{i}_{sig}_{config_num}_delta_{pdf}",setupATLAS=False)
                    u.submit_job(fsubname,walltime=walltime[i],io=0,mem=4)

                fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_save_{i}_{sig}_{config_num}.sh"
                setupLines=[f"cd {outdir}"]
                cmd_to_save=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_make_jobs_tar+csv_table.py {A_size} {B_size} {sig} {N_jobs} {remove} {scriptsdir}/config{config_num}.json {outdir}/training_outcomes {pdf} {scriptsdir}/training_outcomes"
                time.sleep(15)
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
