from pathlib import Path
from sys import argv

from frame.cluster.call_scripts import RUN_PYTHON_JOB_SH_ABS_PATH, run_remote_python
from frame.command_line.handle_args import context_controlled_execution
from frame.config_handle import ExecutionContext
from frame.file_structure import OUTPUT_FILE_NAME, RESULTS_DIR, get_relpath_from_local_root
from train.train_config import TrainConfig

SINGLE_TRAIN_RELPATH_FROM_ROOT = get_relpath_from_local_root(Path(__file__).parent.absolute() / "single_train.py")


@context_controlled_execution
def submit_train(context: ExecutionContext) -> None:
    if not isinstance(context.config, TrainConfig):
        raise ValueError(f"Expected TrainConfig, got {context.config.__class__.__name__}")
    
    # Prepare training job
    ## Verify commit hash matching with remote repository
    # is_same_version_as_remote(context)  # todo: this does not work asynchronously

    # todo: job multiplicity should be done inside here

    # Submit training job
    run_remote_python(
        config:=context.config,
        run_python_bash_script_abspath=RUN_PYTHON_JOB_SH_ABS_PATH,
        workdir_at_cluster_abspath=config.cluster__project_root_at_cluster_abspath,
        environment_activation_script_abspath=config.cluster__environment_activation_script_at_cluster_abspath,
        python_script_relpath_from_workdir_at_cluster=SINGLE_TRAIN_RELPATH_FROM_ROOT,
        script_arguments=argv[1:],
        output_file=RESULTS_DIR / context.unique_descriptor / OUTPUT_FILE_NAME,
    )


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


if __name__ == "__main__":
    submit_train()
