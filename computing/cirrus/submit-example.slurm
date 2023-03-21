#!/bin/bash
# (this script must be executable by user)

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=test-job
#SBATCH --time=96:00:00
#SBATCH --nodes=1           # Leave as 1 to run on same node (each node has 18 cores)
#SBATCH --tasks-per-node=1  # Leave as 1
#SBATCH --cpus-per-task=1   # Leave as 1 unless want multiple cores assigned 

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=d422
# We use the "standard" partition as we are running on CPU nodes
#SBATCH --partition=standard
# We use the "standard" QoS as our runtime is less than 4 days
#SBATCH --qos=standard

# Change to the submission directory - use your own username and desired working directory
cd /work/d422/d422/user/
# Launch the job - N.B. path is relative to that set above
srun --cpu-bind=cores ./run-example.sh "$@"
