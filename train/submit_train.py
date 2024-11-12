from argparse import ArgumentParser
from pathlib import Path
from time import sleep

from frame.command_line_args import context_controlled_execution
from frame.config_handle import ExecutionContext, version_controlled_execution_context
from frame.submit import prepare_submit_files_save, submit_save_jobs
from train.train_config import TrainConfig

@context_controlled_execution
def train(context: ExecutionContext) -> None:
    if not isinstance(context.config, TrainConfig):
        raise ValueError("Expected TrainConfig, got {context.config.__class__.__name__}")
    submit_train_job(context.config)


def submit_train_job(train_config: TrainConfig) -> None:
    out_dir = train_config.out_dir
    runtag = train_config.runtag
    jobid = 1  # ad hoc
    
    # setupLines=[f"cd {out_dir}"]
    # fsubname=f"{out_dir}/{runtag}/submit/sub_{jobid}_{i}_{number_of_signals}_{TAU_or_delta}_{sample}_{pdf}.sh"
    # cmd_TAU=f"/usr/local/anaconda/3.8/bin/python {scriptsdir}/new_training.py -j {jobid} -R {train_size} -B {test_size} -s {number_of_signals} -c {config_path} -o {out_dir}/training_outcomes --seed {seed} "
    # cmd_TAU+=f"-t TAU -S {sample} --BDstr Bkg --SDstr Sig --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
    # u.prepare_submit_file(fsubname, setupLines,[cmd_TAU], shortname=f"sub_TAU", setupATLAS=False)
    # u.submit_job(fsubname,walltime=walltime[i],io=0,mem=4)

    # fsubname=f"{out_dir}/{runtag}"+"/submit/"+f"sub_{jobid}_{i}_{number_of_signals}_delta_{sample}_{pdf}.sh"
    # cmd_delta=f"/usr/local/anaconda/3.8/bin/python {train_config.scripts_dir}/new_training.py -j {jobid} -R {train_size} -B {test_size} -s {number_of_signals} -c {config_path} -o {out_dir}/training_outcomes --seed {seed} "
    # cmd_delta+=f"-t delta -S {train_config.train__histogram_analytic_pdf} --BDstr Ref --BAstr Ref+Bkg+Sig --smlpdf {pdf}"
    # u.prepare_submit_file(fsubname, setupLines, [cmd_delta], shortname=f"sub_delta", setupATLAS=False)
    # u.submit_job(fsubname,walltime=train_config.walltime[i],io=0,mem=4)

    # fsubname=f"{out_dir}/{runtag}"+"/submit/"+f"sub_save.sh"
    # setupLines=[f"cd {out_dir}"]
    # cmd_to_save=f"/usr/local/anaconda/3.8/bin/python {train_config.scripts_dir}/new_make_jobs_tar+csv.py"
    # sleep(30)
    # prepare_submit_files_save(fsubname, setupLines, [cmd_to_save], shortname=f"sub_save", setupATLAS=False)
    # N_jobs_wait = train_config.n_jobs * 2
    # submit_save_jobs(fsubname, N_jobs_wait,walltime=train_config.save_walltime, io=0, mem=2)


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
