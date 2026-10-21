#!/usr/bin/env bash
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$app_dir"
if [[ ! -x .venv/bin/python ]]; then
    echo 'Create .venv and install requirements.txt first. See README.md.' >&2
    exit 1
fi
exec .venv/bin/python -m streamlit run app.py "$@"
