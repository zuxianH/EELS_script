"""Run-folder generation and persistent local/Slurm job controls."""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import uuid

from filelock import FileLock

from .config import dump_settings, effective_parameters, validate_settings
from .profiles import render_script, render_worker
from .storage import atomic_json, atomic_text, atomic_yaml, read_yaml

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def now():
    return datetime.now(timezone.utc).isoformat()


def generated_files(settings: dict, folder: Path) -> dict[str, str]:
    return {
        "run.py": "from tacaw_studio.runner import main\nif __name__ == '__main__':\n    main()\n",
        "supervise.py": "from pathlib import Path\nimport sys\nfrom tacaw_studio.jobs import supervise\nsupervise(Path(__file__).resolve().parent, sys.argv[1])\n",
        "run_local.sh": render_script(settings, folder),
        "resume_local.sh": render_script(settings, folder, resume=True),
        "submit.sh": render_script(settings, folder, slurm=True),
        "submit_resume.sh": render_script(settings, folder, slurm=True, resume=True),
        "check.sh": render_script(settings, folder, action="check"),
        "submit_check.sh": render_script(settings, folder, action="check", slurm=True),
        "convert.sh": render_script(settings, folder, action="convert"),
        "submit_convert.sh": render_script(settings, folder, action="convert", slurm=True),
        "worker.sh": render_worker(settings["execution"]["profile"]),
    }


def generate_run(settings: dict) -> Path:
    settings = copy.deepcopy(settings)
    input_root = Path(settings["input_folder"]).expanduser().absolute()
    settings["input_folder"] = str(input_root)
    settings["inputs"]["files"] = [str((Path(f).expanduser() if Path(f).expanduser().is_absolute()
                                      else input_root/Path(f).expanduser()).absolute())
                                     for f in settings["inputs"]["files"]]
    structure = settings["parameters"].get("sample_structure_file")
    if structure:
        path = Path(structure).expanduser()
        settings["parameters"]["sample_structure_file"] = str(path if path.is_absolute() else input_root/path)
    validate_settings(settings, require_files=True)
    for f in settings["inputs"]["files"]:
        if not Path(f).expanduser().is_file():
            raise ValueError(f"Trajectory does not exist: {f}")
    root = Path(settings.get("output_root") or Path(settings["input_folder"])/"tacaw_runs").expanduser().absolute()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", settings["run_name"]).strip("._") or "tacaw"
    folder = root/f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder/"logs").mkdir()
    try:
        atomic_text(folder/"studio.yaml", dump_settings(settings))
        atomic_yaml(folder/"profile.yaml", settings["execution"]["profile"])
        atomic_yaml(folder/"parameters.yaml", effective_parameters(settings))
        package = Path(__file__).parent
        shutil.copytree(package, folder/"tacaw_studio", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        for name, text in generated_files(settings, folder).items():
            p = folder/name
            atomic_text(p, text)
            if name.endswith(".sh"):
                p.chmod(0o750)
        atomic_json(folder/"lifecycle.json", {"state": "generated", "updated": now()})
        atomic_text(folder/"README_run.txt",
                    "This folder is a frozen Studio run.\n"
                    "run_local.sh / submit.sh prepare trajectories and create the result store ONCE.\n"
                    "resume_local.sh / submit_resume.sh reuse the existing store.\n"
                    "check.sh / submit_check.sh check the configured calculation environment.\n"
                    "convert.sh / submit_convert.sh convert completed results to NumPy.\n"
                    "Keep source trajectories and the configured simulation environment available.\n"
                    "Read logs and lifecycle.json for preparation failures. Use a new folder after incomplete preparation.\n")
    except BaseException:
        # Keep the folder as evidence, never silently remove user run artifacts.
        atomic_json(folder/"lifecycle.json", {"state": "generation_failed", "updated": now()})
        raise
    return folder


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def pid_identity(pid: int) -> str | None:
    # Linux start-time ticks identify a process even when its PID is recycled.
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        parts = text[text.rfind(")")+2:].split()
        if parts[0] == "Z":
            return None
        return parts[19]
    except FileNotFoundError:
        return None
    except OSError:
        return None


def local_alive(job: dict) -> bool:
    pid = job.get("pid")
    if not pid:
        return False
    if Path("/proc").is_dir():
        identity = pid_identity(pid)
        return identity is not None and identity == job.get("pid_identity")
    try:
        os.kill(pid, 0)
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, timeout=5).stdout
        return job.get("supervisor_path", "") in command
    except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
        return False


def slurm_state(job: dict) -> str:
    cluster = ["-M", job["cluster"]] if job.get("cluster") else []
    try:
        result = subprocess.run(["squeue", *cluster, "-h", "-j", job["job_id"], "-o", "%T"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode:
            return "UNKNOWN"
        if result.stdout.strip():
            return result.stdout.strip().splitlines()[0].split()[0]
        result = subprocess.run(["sacct", *cluster, "-j", job["job_id"], "-X", "-n",
                                 "--format=State", "--parsable2"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].split("|")[0].split()[0].rstrip("+")
        # Empty squeue means it is no longer queued; sacct can lag.
        return "ENDED"
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN"


def status(folder: Path) -> dict:
    job = read_json(folder/"job.json")
    lifecycle = read_json(folder/"lifecycle.json")
    active = False
    state = lifecycle.get("state", "unknown")
    if job.get("kind") == "local":
        active = local_alive(job)
        state = "RUNNING" if active else job.get("state", state)
        if not active and state in ("starting", "running"):
            state = "INTERRUPTED"
    elif job.get("kind") == "slurm":
        state = slurm_state(job)
        active = state not in TERMINAL | {"ENDED"}
    result = {"job": job, "state": state, "active": active, "lifecycle": lifecycle}
    config = folder/"scan_gpu_config.yaml"
    if config.exists():
        try:
            values = read_yaml(config)["computation_batches"]["status_list"]
            result["batches"] = {"total": len(values), "finished": values.count(2),
                                 "running": values.count(1), "waiting": values.count(0)}
        except (ValueError, KeyError, OSError):
            pass
    return result


def start(folder: Path, action="run", *, resume=False) -> dict:
    settings = read_yaml(folder/"studio.yaml")
    scheduler = settings["execution"]["profile"]["scheduler"]
    with FileLock(str(folder/".launch.lock"), timeout=1):
        if status(folder)["active"]:
            raise ValueError("A job is already active or its scheduler state is unknown.")
        if action == "run":
            c = folder/"scan_gpu_config.yaml"
            if resume and not c.is_file():
                raise ValueError("Resume requires an existing TACAW config.")
            if not resume and (c.exists() or (folder/"trajectories").exists()):
                raise ValueError("This run was already prepared or partially prepared. Resume it, or generate a new folder.")
            script = ("submit_resume.sh" if resume else "submit.sh") if scheduler == "slurm" else (
                "resume_local.sh" if resume else "run_local.sh")
        else:
            script = ("submit_" if scheduler == "slurm" else "") + ("check.sh" if action == "check" else "convert.sh")
            if action == "convert":
                from .storage import require_complete
                require_complete(folder/"scan_gpu_config.yaml")
        job = {"kind": scheduler, "action": action, "resume": resume, "created": now()}
        if scheduler == "slurm":
            command = ["sbatch", "--parsable", script]
            try:
                result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired as exc:
                raise ValueError("sbatch timed out. Check the queue before retrying; submission may have succeeded.") from exc
            if result.returncode:
                raise ValueError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")
            match = re.fullmatch(r"(\d+)(?:;([\w.-]+))?", result.stdout.strip())
            if not match:
                raise ValueError(f"Unexpected sbatch response: {result.stdout!r}. Check the queue before retrying.")
            job.update(job_id=match[1], cluster=match[2], state="submitted")
            atomic_json(folder/"job.json", job)
        else:
            supervisor = folder/"supervise.py"
            with (folder/"logs"/"launcher.log").open("a") as log:
                process = subprocess.Popen([sys.executable, str(supervisor), script], cwd=folder,
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            job.update(pid=process.pid, pid_identity=pid_identity(process.pid), state="starting",
                       supervisor_path=str(supervisor))
            atomic_json(folder/"job.json", job)
        return job


def supervise(folder: Path, script: str):
    # Parent holds .launch.lock while recording the PID; this lock makes
    # supervisor metadata updates start after the parent has committed it.
    with FileLock(str(folder/".launch.lock")):
        job = read_json(folder/"job.json")
        job["state"] = "running"
        atomic_json(folder/"job.json", job)
    try:
        result = subprocess.run(["bash", str(folder/script)], cwd=folder)
        state = "complete" if result.returncode == 0 else "failed"
        with FileLock(str(folder/".launch.lock")):
            job = read_json(folder/"job.json")
            job.update(state=state, exit_code=result.returncode, ended=now())
            atomic_json(folder/"job.json", job)
        if result.returncode:
            previous = read_json(folder/"lifecycle.json")
            atomic_json(folder/"lifecycle.json", {"state": "failed", "updated": now(),
                        "error": previous.get("error", "Launch failed. See logs/launcher.log."),
                        "action": job["action"]})
    except Exception as exc:
        atomic_json(folder/"lifecycle.json", {"state": "failed", "updated": now(), "error": str(exc)})
        raise


def cancel(folder: Path):
    with FileLock(str(folder/".launch.lock"), timeout=1):
        report = status(folder)
        if not report["active"]:
            raise ValueError("No active job to cancel.")
        job = report["job"]
        if job["kind"] == "slurm":
            cluster = ["-M", job["cluster"]] if job.get("cluster") else []
            try:
                result = subprocess.run(["scancel", *cluster, job["job_id"]], capture_output=True, text=True, timeout=15)
            except subprocess.TimeoutExpired as exc:
                raise ValueError("scancel timed out. Check scheduler status before retrying.") from exc
            if result.returncode:
                raise ValueError(f"scancel failed: {result.stderr}")
            job["state"] = "cancellation_requested"
        else:
            # Verify process identity immediately before signalling its group.
            if not local_alive(job):
                raise ValueError("Local supervisor has already ended.")
            os.killpg(job["pid"], signal.SIGTERM)
            job["state"] = "cancelled"
            atomic_json(folder/"lifecycle.json", {"state": "cancelled", "updated": now()})
        atomic_json(folder/"job.json", job)


def tail(path: Path, limit=16000) -> str:
    with path.open("rb") as h:
        h.seek(0, 2)
        length = h.tell()
        h.seek(max(0, length-limit))
        return h.read(limit).decode("utf-8", errors="replace")
