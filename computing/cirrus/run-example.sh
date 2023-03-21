#!/bin/bash -i
# (start an interactive shell)
# (this script must be executable by user)
# Initialise miniconda
source /work/d422/d422/pfw1/miniconda-init.sh
# load the required python environment
conda activate oqupy
# Additional arguments to sbatch are passed on here
python ./example.py "$@"
# note location of "./" was set in submit-example.slurm

