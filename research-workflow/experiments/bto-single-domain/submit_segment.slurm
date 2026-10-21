#!/bin/bash
# Supply --account and --chdir explicitly when submitting this portable template.
#SBATCH --job-name=bto_long_nvt
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=2-12:00:00
set -euo pipefail
run_dir=$1
target_step=$2
# Frozen manifest selects the installed runtime; do not mix module installations.
runtime=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime"])' "$run_dir/run_manifest.json")
plugin=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["plugin"])' "$run_dir/run_manifest.json")
export PATH="$runtime:$PATH"
export DP_LMP_PLUGIN="$plugin"
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export DP_INTRA_OP_PARALLELISM_THREADS="$OMP_NUM_THREADS"
export DP_INTER_OP_PARALLELISM_THREADS=1
export TF_NUM_INTRAOP_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTEROP_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=2
python "$run_dir/source/segmented_nvt.py" run --output "$run_dir" --until "$target_step"
