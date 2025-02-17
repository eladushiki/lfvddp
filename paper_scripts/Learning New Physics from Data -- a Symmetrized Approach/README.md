# LFVNN-symmetrized

The files used in our paper (https://arxiv.org/abs/2401.09530) are here in paper scripts under the directory with the paper title.
Note: this folder contains legacy implementation. While it was left untouched, much of the functionality here was imported to other files in a reusable manner.

## train
	
`new_submit_(inbar_*).py` - submitting jobs for specific scenario.

Make sure to change the paths for outdir (where you want the output of the training to be saved) and scriptsdir (where the python code to run is), and append the path to the configs if you're using them. Also need access to "/storage/agrp/yuvalzu/mattiasdata" on the cluster.
 		
`new_training.py` - script for trainning. You may also run `new_training_copy.py`.
	 
`new_setting.py` - preparing the datasets input to the model for training. You may also use `new_setting_copy.py`.
		 
`new_make_jobs_tar+csv.py` - compressing h5 files (history and weights) and producing a csv summary with t values at the end of training. You can also try save_jobs_script instead.

needs - /srv01/agrp/yuvalzu/scripts/terminal_scripts/copy_txt_to_csv.py -  I no longer have access to this file, but it should still be on the cluster.
            
## analyze

`fix_plots_for_paper.ipynb` - the main notebook for producing plots. There are a few more in extra_plots_yuval it seems like.
`new_analysis_utils.py` - analyzing the saved files. You may also use `new_analysis_utils_copy.py`.
`new_plot_utils.py` - for plotting.

Make sure to change the paths for dir (where the output of the training was saved) and plots_dir (where you want to save the plots). Also need access to "/storage/agrp/yuvalzu/mattiasdata" on the cluster.

# General things

Mcoll - e-\mu data, exp - exp background with various signals, NPLM - just the method from https://arxiv.org/abs/1806.02350, resample - premutation tests.
Things you probably don't need - anything with _crossentropy.

# table scripts:

This is the new code Yuval wrote. Should be much neater, but I haven't tested it myself.
