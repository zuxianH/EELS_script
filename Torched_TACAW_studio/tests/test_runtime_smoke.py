"""Optional real CPU execution with external Torch/pyms/engine installation."""
import os
from pathlib import Path
import subprocess
import signal

import numpy as np
import pytest
import zarr

from tacaw_studio.jobs import generate_run
from tacaw_studio.python_config import apply_python_config, render_python_config
from tacaw_studio.storage import require_complete


def run_smoke_script(script):
    # Cold imports on a shared cluster filesystem can dominate this tiny run.
    timeout = float(os.environ.get("TACAW_SMOKE_TIMEOUT", "360"))
    process = subprocess.Popen(["bash", str(script)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        pytest.fail(f"CPU smoke test exceeded {timeout:g}s; worker group stopped.\n{stdout}\n{stderr}")
    assert process.returncode == 0, stdout + "\n" + stderr


@pytest.mark.parametrize("kind",["Plane wave","Line scan","2D scan"])
def test_actual_cpu_runtime(settings,kind):
    python = os.environ.get("TACAW_SMOKE_PYTHON")
    engine = os.environ.get("TACAW_SMOKE_ENGINE")
    if not python or not engine:
        pytest.skip("Set TACAW_SMOKE_PYTHON and TACAW_SMOKE_ENGINE for a tiny real calculation.")
    settings["parameters"]["trajectory_chunks_overlap"] = 1.
    settings["scan"].update(kind=kind, start=[1.,1.],end=[2.,2.],
                            shape=[3,1] if kind == "Line scan" else [2,2],batch=[1,1])
    settings["auto_npy"] = True
    settings["execution"]["profile"].update(python=[python],pythonpath=[engine])
    settings = apply_python_config(settings, render_python_config(settings))
    folder = generate_run(settings)
    run_smoke_script(folder / "run_local.sh")
    config = require_complete(folder/"scan_gpu_config.yaml")
    output=Path(config["storage"]["intensities_zarray"])
    expected=np.asarray(zarr.open_array(str(output),mode="r")[:])
    actual=np.load(output.with_suffix(".npy"))
    np.testing.assert_array_equal(actual,expected)
    assert np.isfinite(actual).all()
    coordinates=np.load(folder/"scan_coordinates.npy")
    if kind != "Plane wave":
        np.testing.assert_allclose(coordinates[-1,-1],[2,2])
    # Resuming a completed calculation must preserve values and reuse the store.
    before=expected.copy()
    run_smoke_script(folder / "resume_local.sh")
    np.testing.assert_array_equal(zarr.open_array(str(output),mode="r")[:],before)
