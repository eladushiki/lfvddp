# LFVNN-symmetrized

The files used in our paper (https://arxiv.org/abs/2401.09530) are in paper scripts.

  train:
      new_submit_(inbar_*) - submitting jobs for specific scenario. Make sure to change the paths for outdir (where you want the output of the training to be saved) and scriptsdir (where the python code to run is).
        new_training  - script for trainning. You may also run new_training_copy.
          new_setting - preparing the datasets input to the model for training. You may also use new_setting_copy.
          new_make_jobs_tar+csv - compressing h5 files (history and weights) and producing a csv summary with t values at the end of training. You can also try save_jobs_script instead.
            needs - 
              /srv01/agrp/yuvalzu/scripts/terminal_scripts/copy_txt_to_csv.py -  I no longer have access to this file, but it should still be on the cluster.
            
      
