#!/bin/zsh

#PBS -j oe
#PBS -q N
#PBS -N SymmetryDDPPythonjob

echo "Starting on `hostname`, `date`"
echo "jobs id: ${PBS_JOBID}"

# cd to the required directory
cd $WORKDIR

# run python script with arguments

python $SCRIPT_RELPATH $PYTHON_ARGS

echo "Done, `date`"