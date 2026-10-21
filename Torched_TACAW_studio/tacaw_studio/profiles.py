"""Editable execution profiles and portable shell/Slurm script generation."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import re
import shlex
import sys


def default_profile() -> dict:
    source = os.environ.get("TACAW_SOURCE", "")
    if not source:
        candidate = Path(__file__).resolve().parents[3] / "torched-tacaw-main"
        source = str(candidate) if candidate.is_dir() else ""
    return {
        "name": "Local CPU", "scheduler": "local", "backend": "cpu",
        "python": [sys.executable], "launcher": [], "modules": [],
        "setup_commands": "", "pythonpath": [source] if source else [],
        "environment": {"PYTHONUNBUFFERED": "1"}, "worker_binding": "scheduler",
        "srun_args": [],
    }


def default_slurm() -> dict:
    return {"account": "", "partition": "", "qos": "", "time": "01:00:00",
            "nodes": 1, "tasks_per_node": 1, "cpus_per_task": 1,
            "memory": "", "gpus_per_node": 1,
            "output": "logs/slurm-%x-%j.out", "error": "logs/slurm-%x-%j.err",
            "additional": {}}


def profiles() -> dict[str, dict]:
    base = default_profile()
    out = {"Local CPU": base}
    root = Path(__file__).resolve().parents[3]
    workspace_python = root/"venvs"/"deepmd-ipi"/"bin"/"python"
    if workspace_python.is_file() and (root/"torched-tacaw-main").is_dir():
        out["Workspace CPU"] = dict(copy.deepcopy(base), name="Workspace CPU",
                                   python=[str(workspace_python)],
                                   pythonpath=[str(root/"torched-tacaw-main")])
    for name, backend in (("Local CUDA", "cuda"), ("Local ROCm", "rocm")):
        out[name] = dict(copy.deepcopy(base), name=name, backend=backend)
    out["Generic Slurm"] = dict(copy.deepcopy(base), name="Generic Slurm", scheduler="slurm",
                               backend="cuda", python=["python3"])
    out["LUMI"] = dict(copy.deepcopy(base), name="LUMI", scheduler="slurm", backend="rocm",
                      python=["python"], launcher=["singularity", "exec", "--rocm", "${SIFPYTORCH}"],
                      modules=["LUMI/25.03", "partition/G",
                               "PyTorch/20240918-rocm-6.2.1-python-3.12-singularity-20241007"],
                      worker_binding="rocr-local-rank",
                      environment={"PYTHONUNBUFFERED": "1"})
    out["Dardel"] = dict(copy.deepcopy(base), name="Dardel", scheduler="slurm", backend="rocm",
                        python=["python3"], modules=["PDC", "singularity/4.2.0-cpeGNU-24.11"],
                        launcher=["singularity", "exec", "--rocm", "-B", "/cfs/klemming",
                                  "/pdc/software/resources/sing_hub/rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0"],
                        worker_binding="rocr-local-rank")
    return out


def validate_profile(p: dict):
    if p.get("scheduler") not in ("local", "slurm") or p.get("backend") not in ("cpu", "cuda", "rocm"):
        raise ValueError("Profile needs a local/slurm scheduler and cpu/cuda/rocm backend.")
    for key in ("python", "launcher", "modules", "pythonpath", "srun_args"):
        if not isinstance(p.get(key), list) or any(not isinstance(v, str) or "\n" in v or "\r" in v or "\x00" in v for v in p[key]):
            raise ValueError(f"Profile {key} must be a list of single-line strings.")
    if not p["python"]:
        raise ValueError("Set the simulation Python executable.")
    if not isinstance(p.get("environment"), dict):
        raise ValueError("Environment must be a mapping.")
    for key, value in p["environment"].items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in ("HOME", "CODEX_HOME", "BASH_ENV", "ENV"):
            raise ValueError(f"Invalid or reserved environment name: {key}")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("Environment values must be strings without NUL characters.")
    if p.get("worker_binding") not in ("scheduler", "local-rank", "rocr-local-rank"):
        raise ValueError("Unknown worker binding strategy.")
    if not isinstance(p.get("setup_commands"), str):
        raise ValueError("Setup commands must be text.")


def validate_slurm(s: dict, backend: str):
    for key in ("nodes", "tasks_per_node", "cpus_per_task"):
        if type(s.get(key)) is not int or s[key] < 1:
            raise ValueError(f"Slurm {key} must be a positive integer.")
    if type(s.get("gpus_per_node")) is not int or s["gpus_per_node"] < 0:
        raise ValueError("GPU request must be a nonnegative integer.")
    if backend != "cpu" and s["gpus_per_node"] != s["tasks_per_node"]:
        raise ValueError("Use exactly one task per requested GPU.")
    for key in ("account", "partition", "qos", "time", "memory", "output", "error"):
        v = s.get(key)
        if not isinstance(v, str) or any(c in v for c in ("\n", "\r", "\x00")):
            raise ValueError(f"Slurm {key} must be a single-line string.")
    if not re.fullmatch(r"(?:\d+-)?\d{1,3}:\d{2}:\d{2}", s["time"]):
        raise ValueError("Slurm time must be HH:MM:SS or D-HH:MM:SS.")
    if not isinstance(s.get("additional"), dict):
        raise ValueError("Additional SBATCH directives must be a mapping.")
    reserved = {"account", "partition", "qos", "time", "nodes", "ntasks", "ntasks-per-node",
                "cpus-per-task", "mem", "mem-per-cpu", "gpus", "gpus-per-node",
                "gpus-per-task", "job-name", "output", "error", "chdir", "wrap"}
    for key, val in s["additional"].items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", key) or key in reserved:
            raise ValueError(f"Invalid, conflicting, or reserved additional SBATCH option: {key}")
        if val is not None and (not isinstance(val, (str, int)) or
                                any(c in str(val) for c in ("\n", "\r", "\x00"))):
            raise ValueError(f"Invalid SBATCH value: {key}")


def shell_token(value: str) -> str:
    # An entire token may explicitly reference an environment variable; never
    # interpolate arbitrary shell syntax embedded in a profile path.
    if re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", value):
        return '"' + value + '"'
    return shlex.quote(value)


def shell_setup(profile: dict) -> list[str]:
    lines = []
    if profile["modules"]:
        lines.append("module load " + " ".join(shlex.quote(x) for x in profile["modules"]))
    if profile["setup_commands"].strip():
        lines.append(profile["setup_commands"])
    for key, value in profile["environment"].items():
        lines.append(f"export {key}={shlex.quote(value)}")
    if profile["pythonpath"]:
        pp = ":".join(profile["pythonpath"])
        lines.append("export PYTHONPATH=" + shlex.quote(pp) + '"${PYTHONPATH:+:$PYTHONPATH}"')
    if "OMP_NUM_THREADS" not in profile["environment"]:
        lines.append('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-1}}"')
    command = " ".join(shell_token(x) for x in profile["launcher"] + profile["python"])
    # Environment variables set on the host are explicitly forwarded into
    # containers, whose own Python must be used for dependency checks.
    if profile["launcher"]:
        command = " ".join(shell_token(x) for x in profile["launcher"])
        command += ' env -u BASH_ENV -u ENV -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL "PYTHONPATH=${PYTHONPATH:-}" "OMP_NUM_THREADS=$OMP_NUM_THREADS"'
        for key in profile["environment"]:
            command += f' "{key}=${key}"'
        command += ' "${tacaw_gpu_env[@]}"'
        command += " " + " ".join(shell_token(x) for x in profile["python"])
    if profile["launcher"]:
        lines.append('run_python() { local -a tacaw_gpu_env=(); local tacaw_gpu_var; '
                     'for tacaw_gpu_var in CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL; '
                     'do if printenv "$tacaw_gpu_var" >/dev/null; then tacaw_gpu_env+=("$tacaw_gpu_var=${!tacaw_gpu_var}"); fi; done; '
                     + command + ' "$@"; }')
    else:
        lines.append(f'run_python() {{ {command} "$@"; }}')
    return lines


def render_script(settings: dict, run_folder: Path, action="run", *, slurm=False, resume=False) -> str:
    p = settings["execution"]["profile"]
    s = settings["execution"]["slurm"]
    validate_profile(p)
    validate_slurm(s, p["backend"])
    lines = ["#!/usr/bin/env bash"]
    if slurm:
        options = {"job-name": re.sub(r"[^A-Za-z0-9_.-]+", "_", settings["run_name"])[:100],
                   "chdir": str(run_folder),
                   "account": s["account"], "partition": s["partition"], "qos": s["qos"],
                   "time": s["time"], "nodes": s["nodes"],
                   "ntasks-per-node": s["tasks_per_node"], "cpus-per-task": s["cpus_per_task"],
                   "mem": s["memory"], "output": s["output"], "error": s["error"]}
        if p["backend"] != "cpu":
            options["gpus-per-node"] = s["gpus_per_node"]
        options.update(s["additional"])
        for key, value in options.items():
            if value is None:
                lines.append(f"#SBATCH --{key}")
            elif value != "":
                lines.append(f"#SBATCH --{key}={shlex.quote(str(value))}")
    lines += ["set -euo pipefail", 'cd -- "$(dirname -- "${BASH_SOURCE[0]}")"', 'mkdir -p logs']
    lines += shell_setup(p)
    lines += ['runner="$PWD/run.py"']
    if action == "check":
        lines.append('run_python "$runner" check --backend ' + p["backend"])
    elif action == "convert":
        lines.append('run_python "$runner" convert')
    else:
        lines.append('run_python "$runner" check --backend ' + p["backend"])
        lines.append('run_python "$runner" ' + ("resume" if resume else "prepare"))
        if slurm:
            srun = ["srun", "--kill-on-bad-exit=1"]
            if p["backend"] != "cpu":
                if p["worker_binding"] == "scheduler":
                    srun += ["--gpus-per-task=1", "--gpu-bind=single:1"]
                else:
                    srun += ["--gpu-bind=none"]
            srun += p["srun_args"]
            launch = " ".join(shlex.quote(v) for v in srun)
            # worker.sh performs environment setup inside each srun task.
            lines.append(launch + ' bash "$PWD/worker.sh"')
        else:
            lines.append('run_python "$runner" worker --backend ' + p["backend"] +
                         ' --binding ' + p["worker_binding"])
        lines.append('run_python "$runner" finish')
    return "\n".join(lines) + "\n"


def render_worker(profile: dict) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail",
             'cd -- "$(dirname -- "${BASH_SOURCE[0]}")"']
    lines += shell_setup(profile)
    if profile["backend"] == "rocm" and profile["worker_binding"] == "rocr-local-rank":
        lines += ["unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES GPU_DEVICE_ORDINAL",
                  'export ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:-0}"']
    lines.append('run_python "$PWD/run.py" worker --backend ' + profile["backend"] +
                 ' --binding ' + profile["worker_binding"])
    return "\n".join(lines) + "\n"
