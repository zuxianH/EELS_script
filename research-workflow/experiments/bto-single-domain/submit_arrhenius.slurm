#!/bin/bash
#SBATCH --job-name=bto_single_domain
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=2-12:00:00
#SBATCH --output=bto-%j.out

# One MPI task using the wheel's matched MPICH; TF/OpenMP uses the allocated
# CPU threads. Do not mix this environment with another MPI/LAMMPS module.
set -euo pipefail
# Submit from this experiment directory, with an explicit --account argument.
cd "${BTO_EXPERIMENT_DIR:-${SLURM_SUBMIT_DIR:?}}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export DP_INTRA_OP_PARALLELISM_THREADS="$SLURM_CPUS_PER_TASK"
export DP_INTER_OP_PARALLELISM_THREADS=1
source ./env.sh
python run_workflow.py "$@"
