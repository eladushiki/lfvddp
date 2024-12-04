#!/bin/zsh

#PBS -j oe
#PBS -q N
#PBS -N SymmetryDDPPythonjob
#PBS -o SymmetryDDPPythonjob.out
#PBS -e SymmetryDDPPythonjob.err

echo "Starting on `hostname`, `date`"
echo "jobs id: ${PBS_JOBID}"

# cd to the required directory
cd $1

# load python
module load /usr/bin/python

# run python script with the rest of the arguments
shift 1
python $@

echo "Done, `date`"