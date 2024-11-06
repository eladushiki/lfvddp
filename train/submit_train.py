from argparse import ArgumentParser
from pathlib import Path

from frame.config_handle import version_controlled_execution_context
from train.train_config import TrainConfig

def train(cluster_config_path: Path, train_config_path: Path) -> None:
    config = TrainConfig.load_from_files([cluster_config_path, train_config_path])
    with version_controlled_execution_context(config) as context:
        submit_train_job(config)  # todo: how to differentiate when running train from other stuff here?

def submit_train_job(train_config: TrainConfig) -> None:
    out_dir = train_config.out_dir
    runtag = train_config.runtag
    
    setupLines=[f"cd {out_dir}"]
    fsubname=f"{out_dir}/{runtag}/submit/sub_{jobid}_{i}_{number_of_signals}_{TAU_or_delta}_{sample}_{pdf}.sh"
    cmd_TAU=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_training.py -j {jobid} -R {train_size} -B {test_size} -s {number_of_signals} -c {config_path} -o {outdir}/training_outcomes --seed {seed} "
    cmd_TAU+=f"-t TAU -S {sample} --BDstr Bkg --SDstr Sig --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
    u.prepare_submit_file(fsubname,setupLines,[cmd_TAU],shortname=f"sub_{jobid}_{i}_{number_of_signals}_TAU_{sample}_{pdf}",setupATLAS=False)
    u.submit_job(fsubname,walltime=walltime[i],io=0,mem=4)

    fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_{jobid}_{i}_{number_of_signals}_delta_{sample}_{pdf}.sh"
    cmd_delta=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_training.py -j {jobid} -R {train_size} -B {test_size} -s {number_of_signals} -c {config_path} -o {outdir}/training_outcomes --seed {seed} "
    cmd_delta+=f"-t delta -S {sample} --BDstr Ref --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
    u.prepare_submit_file(fsubname,setupLines,[cmd_delta],shortname=f"sub_{jobid}_{i}_{number_of_signals}_delta_{sample}_{pdf}",setupATLAS=False)
    u.submit_job(fsubname,walltime=walltime[i],io=0,mem=4)

    fsubname=f"{outdir}/{runtag}"+"/submit/"+f"sub_save_{i}_{number_of_signals}.sh"
    setupLines=[f"cd {outdir}"]
    cmd_to_save=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_make_jobs_tar+csv.py {train_size} {test_size} {number_of_signals} {N_jobs} {remove} {config_path} {outdir}/training_outcomes {sample} {pdf} {scriptsdir}/training_outcomes"
    time.sleep(30)
    jobs.prepare_submit_files_save(fsubname,setupLines,[cmd_to_save],shortname=f"sub_save_{i}_{number_of_signals}",setupATLAS=False)
    N_jobs_wait = N_jobs+N_jobs
    jobs.submit_save_jobs(fsubname,N_jobs_wait,walltime=save_walltime,io=0,mem=2)


def main():
    parser = ArgumentParser()

    # Arguments for running training
    parser.add_argument(
        "--cluster-config", type=Path, required=True,
        help="Path to cluster configuration file", dest="cluster_config_path"
    )
    parser.add_argument(
        "--train-config", type=Path, required=True,
        help="Path to training configuration file", dest="train_config_path"
    )

    args = parser.parse_args()

    # Call main with configuration paths
    train(args.cluster_config_path, args.train_config_path)

if __name__ == "__main__":
    main()
