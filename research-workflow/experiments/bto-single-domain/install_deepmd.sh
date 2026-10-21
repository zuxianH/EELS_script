#!/bin/bash
# Reproduce the isolated CPU execution environment in ${BTO_SOFTWARE_ROOT:-.software}.
# No system installation and no modification of ~/.bashrc.
set -euo pipefail
bto_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
software_root="${BTO_SOFTWARE_ROOT:-$bto_root/.software}"
mkdir -p "$software_root"
python3 -m venv "$software_root/bootstrap"
"$software_root/bootstrap/bin/python" -m pip install --no-cache-dir uv==0.13.4
export UV_CACHE_DIR="$software_root/cache"
export UV_PYTHON_INSTALL_DIR="$software_root/python"
uv_bin="$software_root/bootstrap/bin/uv"
if [[ ! -x "$software_root/deepmd/bin/python" ]]; then
    "$uv_bin" venv --python 3.11 "$software_root/deepmd"
fi
if [[ -f "$bto_root/requirements-runtime.lock" ]]; then
    "$uv_bin" pip sync --python "$software_root/deepmd/bin/python" "$bto_root/requirements-runtime.lock"
else
    "$uv_bin" pip install --python "$software_root/deepmd/bin/python" \
        'deepmd-kit[cpu,lmp]==3.2.0' 'ase>=3.23,<4' nvidia-cuda-runtime-cu12
fi
printf 'Installed. Activate with: source %s/env.sh\n' "$bto_root"
