import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import zarr

from tacaw_studio import jobs
from tacaw_studio.profiles import render_script, render_worker, validate_slurm
from tacaw_studio.storage import copy_to_npy, inspect_array, require_complete, atomic_yaml
from tacaw_studio.runner import chunk_starts


def test_chunk_count_matches_engine_boundary(settings):
    p = settings["parameters"]
    assert chunk_starts(p, 12) == [0]
    p["trajectory_chunks_nof"] = None
    p["trajectory_chunks_overlap"] = 1.
    assert chunk_starts(p, 12) == [0,4]
    with pytest.raises(ValueError, match="No TACAW chunks"):
        chunk_starts(p, 4)


def test_bundle_and_shell_syntax(settings):
    settings["run_name"] = "a case with spaces"
    folder = jobs.generate_run(settings)
    assert (folder/"studio.yaml").exists()
    assert (folder/"tacaw_studio"/"runner.py").exists()
    assert not (folder/"scan_gpu_config.yaml").exists()
    assert not (folder/"scan_gpu_tacaw.zarray").exists()
    for script in folder.glob("*.sh"):
        result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    shell = (folder/"submit.sh").read_text()
    assert shell.index("prepare") < shell.index("srun") < shell.index("finish")
    resume = (folder/"submit_resume.sh").read_text()
    assert '"$runner" resume' in resume
    assert '"$runner" prepare' not in resume
    assert (folder/"worker.sh").read_text().count("worker --backend") == 1


def test_script_quotes_literal_paths_and_env(settings, tmp_path):
    probe = tmp_path/"python probe.py"
    output = tmp_path/"argv.json"
    probe.write_text("import json,sys,os\nfrom pathlib import Path\nPath(os.environ['PROBE_OUT']).write_text(json.dumps({'argv':sys.argv[1:],'value':os.environ['LITERAL']}))\n")
    p = settings["execution"]["profile"]
    p["python"] = [sys.executable, str(probe)]
    p["environment"] = {"PROBE_OUT":str(output), "LITERAL":'literal $(touch unwanted) `touch unwanted` $OTHER "quote"'}
    folder = tmp_path/"run $(touch unwanted)"
    folder.mkdir()
    (folder/"check.sh").write_text(render_script(settings,folder,action="check"))
    result = subprocess.run(["bash", str(folder/"check.sh")], capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    recorded = json.loads(output.read_text())
    assert recorded["value"] == p["environment"]["LITERAL"]
    assert recorded["argv"][-3:] == ["check","--backend","cpu"]
    assert not (folder/"unwanted").exists()
    assert not (tmp_path/"unwanted").exists()


def test_slurm_cpu_gpu_and_directive_validation(settings, tmp_path):
    s = settings["execution"]["slurm"]
    s["additional"] = {"exclusive":None, "constraint":"gpu"}
    script = render_script(settings,tmp_path,slurm=True)
    assert "#SBATCH --exclusive\n" in script
    assert "--gpus-per-node" not in script
    p = settings["execution"]["profile"]
    p["backend"] = "rocm"
    p["worker_binding"] = "rocr-local-rank"
    s.update(nodes=3,tasks_per_node=8,gpus_per_node=8,cpus_per_task=7)
    script = render_script(settings,tmp_path,slurm=True)
    assert "#SBATCH --nodes=3" in script
    assert "--gpu-bind=none" in script
    assert 'ROCR_VISIBLE_DEVICES="${SLURM_LOCALID:-0}"' in render_worker(p)
    s["additional"] = {"ntasks":99}
    with pytest.raises(ValueError, match="reserved"):
        validate_slurm(s,"rocm")


def test_submission_and_duplicate_guard(settings, monkeypatch):
    settings["execution"]["profile"]["scheduler"] = "slurm"
    folder = jobs.generate_run(settings)
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "sbatch":
            return SimpleNamespace(returncode=0,stdout="12345;testcluster\n",stderr="")
        if command[0] == "squeue":
            return SimpleNamespace(returncode=0,stdout="RUNNING\n",stderr="")
        return SimpleNamespace(returncode=0,stdout="",stderr="")
    monkeypatch.setattr(subprocess,"run",fake_run)
    job = jobs.start(folder)
    assert job["job_id"] == "12345"
    assert job["cluster"] == "testcluster"
    with pytest.raises(ValueError, match="already active"):
        jobs.start(folder,resume=True)
    assert sum(c[0] == "sbatch" for c in calls) == 1
    jobs.cancel(folder)
    assert any(c[0] == "scancel" for c in calls)


def test_local_failure_survives_ui_exit(settings, tmp_path):
    # Launch a deliberately failing simulation command, without importing Torch.
    settings["execution"]["profile"]["python"] = [sys.executable, "-c", "raise SystemExit(7)"]
    folder = jobs.generate_run(settings)
    job = jobs.start(folder)
    assert job["kind"] == "local"
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        state = jobs.status(folder)
        if not state["active"] and state["job"].get("exit_code") is not None:
            break
        time.sleep(.1)
    else:
        if jobs.status(folder)["active"]:
            jobs.cancel(folder)
        pytest.fail("Supervisor did not record completion")
    assert state["job"]["exit_code"] == 7
    assert state["lifecycle"]["state"] == "failed"
    assert (folder/"logs"/"launcher.log").is_file()


def test_chunk_copy_axis_order_edge_chunks_and_atomic_failure(tmp_path):
    data = np.arange(1*7*3*2*5*4,dtype=float).reshape(1,7,3,2,5,4)
    array = zarr.open_array(str(tmp_path/"source.zarray"), mode="w", shape=data.shape,
                           chunks=(1,3,2,1,3,2),dtype=data.dtype)
    array[:] = data
    output = tmp_path/"intensity.npy"
    copy_to_npy(array,output)
    np.testing.assert_array_equal(np.load(output),data)
    assert inspect_array(tmp_path/"source.zarray")["bytes"] == data.nbytes
    with pytest.raises(FileExistsError):
        copy_to_npy(array,output)
    def fail(*args,**kwargs):
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError):
        copy_to_npy(array,tmp_path/"failed.npy",report=fail)
    assert not (tmp_path/"failed.npy").exists()
    assert not list(tmp_path.glob("*.partial.npy"))


def test_conversion_requires_all_batches_finished(tmp_path):
    path = tmp_path/"config.yaml"
    for status in ([],[0,2],[1,2]):
        atomic_yaml(path,{"computation_batches":{"status_list":status}})
        with pytest.raises(ValueError,match="incomplete"):
            require_complete(path)
    atomic_yaml(path,{"computation_batches":{"status_list":[2,2]}})
    assert require_complete(path)


def test_relative_input_paths_are_frozen_absolute(settings):
    settings["inputs"]["files"] = ["input.traj"]
    folder = jobs.generate_run(settings)
    from tacaw_studio.storage import read_yaml
    frozen = read_yaml(folder/"studio.yaml")
    assert frozen["inputs"]["files"] == [str(Path(settings["input_folder"])/"input.traj")]
    assert settings["inputs"]["files"] == ["input.traj"]


def test_numpy_reuse_requires_unchanged_recorded_source(tmp_path):
    from tacaw_studio.storage import convert_run
    source = tmp_path/"intensity.zarray"
    array = zarr.open_array(str(source),mode="w",shape=(1,3,2),chunks=(1,2,1),dtype=float)
    array[:] = np.arange(6).reshape(1,3,2)
    atomic_yaml(tmp_path/"scan_gpu_config.yaml",{
        "storage":{"intensities_zarray":str(source)},
        "computation_batches":{"status_list":[2]}})
    npy = source.with_suffix(".npy")
    np.save(npy,np.zeros(array.shape))
    with pytest.raises(FileExistsError,match="no Studio"):
        convert_run(tmp_path)
    npy.unlink()
    assert convert_run(tmp_path) == npy
    assert convert_run(tmp_path) == npy
    array[0,0,0] = 99
    with pytest.raises(ValueError,match="changed"):
        convert_run(tmp_path)
    assert np.load(npy)[0,0,0] == 0


def test_cancel_stops_owned_local_process_group(settings):
    settings["execution"]["profile"]["python"] = [sys.executable,"-c","import time;time.sleep(60)"]
    folder = jobs.generate_run(settings)
    jobs.start(folder)
    deadline = time.monotonic()+15
    while time.monotonic() < deadline:
        report = jobs.status(folder)
        if report["job"].get("state") == "running":
            break
        time.sleep(.1)
    jobs.cancel(folder)
    deadline = time.monotonic()+10
    while time.monotonic() < deadline and jobs.status(folder)["active"]:
        time.sleep(.1)
    assert not jobs.status(folder)["active"]
    assert jobs.status(folder)["job"]["state"] == "cancelled"


def test_container_launcher_forwards_worker_visibility(settings,tmp_path,monkeypatch):
    probe=tmp_path/"probe.py"
    output=tmp_path/"gpu.txt"
    probe.write_text("import os\nfrom pathlib import Path\nPath(os.environ['PROBE_OUT']).write_text(os.environ['ROCR_VISIBLE_DEVICES'])\n")
    profile=settings["execution"]["profile"]
    profile.update(launcher=["bash","-c",'exec "$@"',"_"],python=[sys.executable,str(probe)],
                   environment={"PROBE_OUT":str(output)})
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES","5")
    script=tmp_path/"check.sh"
    script.write_text(render_script(settings,tmp_path,action="check"))
    result=subprocess.run(["bash",str(script)],capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert output.read_text() == "5"
