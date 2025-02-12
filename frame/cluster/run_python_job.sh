#!/bin/zsh

echo "Starting on `hostname`, `date`"
echo "jobs id: ${PBS_JOBID}"

# Check if the job is part of an array, separate output accordingly
if [ -z "$PBS_ARRAYID" ]; then
    echo "This is a singular job and not part of an array."
    OUTPUT_DIR="$OUTPUT_DIR/$PBS_ARRAYID"
else
    echo "Job number within batch is ${PBS_ARRAYID}"
fi

# cd to the required directory

echo "changing to $WORKDIR"
cd $WORKDIR

# making separate output directory for each job in the array
mkdir -p "$OUTPUT_DIR"
exec > "$OUTPUT_DIR/$OUTPUT_FILENAME" 2>&1

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