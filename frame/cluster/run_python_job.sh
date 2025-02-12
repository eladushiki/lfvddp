#!/bin/zsh

echo "Starting on `hostname`, `date`"
echo "jobs id: ${PBS_JOBID}"

# cd to the required directory
echo "changing to $WORKDIR"
cd $WORKDIR

# source python environment
echo "sourcing $ENV_ACTIVATION_SCRIPT"
source $ENV_ACTIVATION_SCRIPT

# Main script, echo on
set -x

# Python info
which python
python --version

# run python script with arguments
eval "python $SCRIPT_RELPATH $PYTHON_ARGS"

echo "Done, $(date)"