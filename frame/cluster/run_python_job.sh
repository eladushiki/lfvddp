#!/bin/zsh

#PBS -j oe
#PBS -q N
#PBS -N SymmetryDDPPythonjob

echo "Starting on `hostname`, `date`"
echo "jobs id: ${PBS_JOBID}"

# cd to the required directory
cd $WORKDIR

# source python environment
source $ENV_ACTIVATION_SCRIPT

# run python script with arguments
python $SCRIPT_RELPATH $PYTHON_ARGS

echo "Done, `date`"