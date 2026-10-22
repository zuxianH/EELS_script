# Source this file before running the BTO workflow. CPU installation, x86-64.
_bto_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
_bto_env="${BTO_SOFTWARE_ROOT:-$_bto_dir/.software}/deepmd"
if [[ ! -x "$_bto_env/bin/python" ]]; then
    printf 'Missing DeePMD environment: %s\n' "$_bto_env" >&2
    return 1
fi
export PATH="$_bto_env/bin:$PATH"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export DP_INTRA_OP_PARALLELISM_THREADS="${DP_INTRA_OP_PARALLELISM_THREADS:-$OMP_NUM_THREADS}"
export DP_INTER_OP_PARALLELISM_THREADS="${DP_INTER_OP_PARALLELISM_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="$DP_INTRA_OP_PARALLELISM_THREADS"
export TF_NUM_INTEROP_THREADS="$DP_INTER_OP_PARALLELISM_THREADS"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
# This environment is for CPU nodes; GPU jobs need a separate installation.
export CUDA_VISIBLE_DEVICES=-1
_bto_plugins=("$_bto_env"/lib/python*/site-packages/deepmd/lib/libdeepmd_lmpplugin.so)
if [[ ! -r "${_bto_plugins[0]}" ]]; then
    printf 'Cannot locate the DeePMD LAMMPS plugin.\n' >&2
    return 1
fi
export DP_LMP_PLUGIN="${_bto_plugins[0]}"
unset _bto_dir _bto_env _bto_plugins
