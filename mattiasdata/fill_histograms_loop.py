import os, sys
import array
import json
from dothestuff import get_input_files

topdir = "/srv01/agrp/yuvalzu/mattiasdata"


input_path="/srv01/agrp/mattiasb/runners/PhenoOutput/pheno-yesbrem20/merge_outputs/"
input_files=get_input_files(input_path,skipZ=True) ## Skipping Drell-Yann else takes very long


for fname in input_files:

   wtime = '30:00'
   mem   = '16g'
   cmd = "qsub -q N -l walltime={},mem={} -v fname={} run_histograms.sh".format(wtime,mem,fname)
   print(cmd)
   os.system(cmd)