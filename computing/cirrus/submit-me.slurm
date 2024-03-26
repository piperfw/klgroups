#!/bin/bash

# Slurm job options 
#SBATCH --job-name=test-job
#SBATCH --time=96:00:00
#SBATCH --nodes=1           # Leave as 1 to run on same node (each node has 18 cores)
#SBATCH --tasks-per-node=1  # Leave as 1
#SBATCH --cpus-per-task=1   # Leave as 1 unless want multiple cores assigned 
#
# Specify project code
#SBATCH --account=d422
# Use the "standard" partition as running on CPU nodes
#SBATCH --partition=standard
# Use the "standard" QoS as our runtime is less than 4 days
#SBATCH --qos=standard

# Change to a directory under /work - use your own username and desired working directory
cd /work/d422/d422/user/

# Load a relevant python module or environment - uncomment ONE of the following lines
module load python/3.9.13 # System python module, no special environment (set version as required)
#source /work/d422/d422/user/my-env # System python module with environment (edit user/my-env as required)
#source /work/d422/d422/user/miniconda-init.sh; conda activate oqupy # Local Miniconda installation

# Avoid Matplotlib warning
#export MPLCONFIGDIR="/work/d422/d422/user/.cache/matplotlib" # EDIT user

# Launch the job with any arguments passed to this script
srun --cpu-bind=cores ./run-me.py "$@"
# N.B. path is relative to that set by cd above and --cpu-bind=cores option should be used
