# LFVNN-symmetrized

## Paper Scripts

Existing file in commit 01017a30 were used to generate the paper (https://arxiv.org/abs/2401.09530).
Relevant files are in a directory that was called then paper scripts.

## Training
	
new_submit_(inbar_*) - submitting jobs for specific scenario. 
Make sure to change the paths for outdir (where you want the output of the training to be saved) and scriptsdir (where the python code to run is), and append the path to the configs if you're using them. Also need access to "/storage/agrp/yuvalzu/mattiasdata" on the cluster.
	 
new_setting - preparing the datasets input to the model for training. You may also use new_setting_copy.
		 
new_make_jobs_tar+csv - compressing h5 files (history and weights) and producing a csv summary with t values at the end of training. You can also try save_jobs_script instead.
needs - /srv01/agrp/yuvalzu/scripts/terminal_scripts/copy_txt_to_csv.py -  I no longer have access to this file, but it should still be on the cluster.
            
## Analysis

fix_plots_for_paper - the main notebook for producing plots. There are a few more in extra_plots_yuval it seems like.
new_analysis_utils - analyzing the saved files. You may also use new_analysis_utils_copy.
new_plot_utils - for plotting.

Make sure to change the paths for dir (where the output of the training was saved) and plots_dir (where you want to save the plots). Also need access to "/storage/agrp/yuvalzu/mattiasdata" on the cluster.

## Table Scripts

This is the new code Yuval wrote. Should be much neater, but Inbar didn't test it herself.

# Additional Remarks

Mcoll - e-\mu data, exp - exp background with various signals, NPLM - just the method from https://arxiv.org/abs/1806.02350, resample - premutation tests.

---
# Installation

It is customary to run

> pip install -e .

or possibly

> pip3 install -e .

to be able to run the code locally while editing it in place.

To also be able to run on the cluster, the same command sould be also run there. The following command might be of use beforehand:

> export PATH="$PATH:/srv01/agrp/<your-username>/.local/bin"

if pip is not already recognized.

To submit SGE jobs to the WIS cluster, you should install slurm. The [specific isntallation command](https://command-not-found.com/qsub) depends on your OS.

# Project configuration
## ATLAS Cluster SSH connection

This might be a tricky part. There is adifferent way to connect to the cluster, whether you're inside WIS_Secure wifi or outside the institute using a VPN.

Starting form the simple part - if you're in the institute you should be able to SSH seamlessly to the cluster using your username and password, after creating one OFC. Contact lcg-managers@weizmann.ac.il for help on this (or any cluster related issue).

There is a more complex procedure if you're using a VPN, which requires you to use a proxy jump, as of the time of writing this. You can contact Elad to save time on configuring this.

Be sure to use different configuration files (explained below) for each case.

# Usage and Expansion Explanation
## Everyting runs in a context

To be able to keep track of every run and its products, a running context is implemented. It does a few things:

- Records the current running configuration
- Records the current Git commit
- To do so reliably, forces you to commit changes before running
- Enables the user to add any additional context as new parameters of the context to be documented
- Documents time of run
- Forces seeding of numpy random, for reproducibility of the results
- Saves it in a file, that should be adjacent to the resulting output

What is not version controlled:

- External databases used, should be version controlled separately (although their locations are).
- That inclueds the version of NN's and such (although their locations are).
- Virtual environment dependency versions are not (yet) well documented.
- Any configuration of external tools. Any scripts in user defined "scripts_dir".

To use this functionality in any new entry point, run the main function inside the context.

**This means that we are not allowed to tamper with git history. Amending or rewriting commits in any other way is prohibited, for it may create results that are non-reproducible!**

# Configuration files
The configuration files and then the Config* dataclasses are the structures that should contain all the parameters that are run-individual.

While Config* classes' contents are divided logically to different classes, the program can be called with any number of `.json` configuration file types, as long as they contain togehter all the necessary parameters. This is implemented so to enable single-file-for-run usage, as well as separation for personal privacy and context-dependent needs.

Is is specifically recommended that personal username and password for SSH connection with the WIS cluster would be stored in a separate file and not added to git. This is why the example `basic_user_config.json` file contains `cluster__*` parameters, which later end up in the `ClusterConfig` dataclass.

## Training

Training entry points:
- `submit_train.py` for remote submission
- `single_train.py` for the server to run each time
