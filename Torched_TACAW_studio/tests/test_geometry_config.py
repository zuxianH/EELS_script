import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from tacaw_studio.config import (defaults, dump_settings, import_config, effective_parameters,
                                preset, validate_parameters, validate_settings)
from tacaw_studio.geometry import coordinates, scan_parameters

@pytest.mark.parametrize("start,end", [([0,0],[3,0]), ([0,0],[0,3]), ([1,2],[4,6]), ([4,6],[1,2])])
def test_exact_line_endpoints(start, end):
    scan = dict(kind="Line scan", start=start, end=end, shape=[7,1], batch=[1,1], periodic=False)
    p = scan_parameters(scan)
    pts = coordinates(p)[:, 0]
    np.testing.assert_allclose(pts[0], start)
    np.testing.assert_allclose(pts[-1], end)
    np.testing.assert_allclose(np.diff(pts, axis=0), np.tile((np.array(end)-start)/6, (6,1)))

def test_2d_rectangle_and_periodic_edges():
    scan = dict(kind="2D scan", start=[3,5], end=[-1,1], shape=[5,3], batch=[1,1], periodic=False)
    pts = coordinates(scan_parameters(scan))
    np.testing.assert_allclose(pts[0,0], [3,5])
    np.testing.assert_allclose(pts[-1,-1], [-1,1])
    scan["periodic"] = True
    pts = coordinates(scan_parameters(scan))
    np.testing.assert_allclose(pts[-1,-1], [3-4*4/5, 5-4*2/3])

def test_tilted_grid_matches_actual_engine():
    scan = dict(kind="Basis vectors", start=[2,-3], basis=[[2,1],[-1,3]],
                shape=[4,6], batch=[2,3])
    p = scan_parameters(scan)
    root = Path(__file__).resolve().parents[3]/"torched-tacaw-main"/"torched_tacaw"/"coordinates.py"
    if not root.exists():
        pytest.skip("Local engine source unavailable")
    spec = importlib.util.spec_from_file_location("engine_coordinates", root)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = module.scanning_grids_coordinates_cartesian(p["scanning_shape"], p["scanning_basis_vectors"], p["scanning_origin"])
    np.testing.assert_allclose(coordinates(p), expected.transpose(1,2,0))

@pytest.mark.parametrize("changes", [{"shape":[1,1]}, {"end":[0,0]}, {"batch":[2,1]}, {"start":[float('nan'),0]}])
def test_invalid_line(changes):
    scan = dict(kind="Line scan", start=[0,0], end=[1,0], shape=[5,1], batch=[1,1], periodic=False)
    scan.update(changes)
    with pytest.raises(ValueError):
        scan_parameters(scan)

def test_studio_round_trip_and_preset_geometry():
    s = defaults()
    s = preset("BTO 2D scan", s)
    result = import_config(dump_settings(s))
    assert result == s
    p = effective_parameters(result)
    np.testing.assert_allclose(p["scanning_basis_vectors"], np.diag([4.023304441*1.5,4.014531257*1.5]))
    assert p["scanning_shape"] == [30,30]
    assert result["scan"]["periodic"]
    for name in ("SiC plane wave", "Si line scan", "STO line scan"):
        validate_settings(preset(name, defaults()))

def test_real_example_config_imports():
    root = Path(__file__).resolve().parents[3]/"torched_TACAW"
    names = ["SiC/tacaw_results_planewave/scan_gpu_config.yaml",
             "Si_Langevin_300K/tacaw_results/scan_gpu_config.yaml",
             "BTO_Ba-O_bussi/tacaw_results_stemeels_2Dscan/scan_gpu_config.yaml",
             "STO_Ti-O_bussi/tacaw_results_stemeels/scan_gpu_config.yaml"]
    if not root.exists():
        pytest.skip("Local examples unavailable")
    for name in names:
        original = yaml.safe_load((root/name).read_text())
        s = import_config((root/name).read_text())
        p = effective_parameters(s)
        assert p["scanning_shape"] == original["beam"]["scanning"]["shape"]
        assert p["trajectory_chunks_nof"] == original["trajectory"]["chunks"]["nof"]
        assert "computation_batches" not in p
        validate_settings(s)

@pytest.mark.parametrize("key,value", [("trajectory_chunks_overlap",1e9), ("frequency_THz_ROI",[1,1]),
                                       ("kspace_ROI_min",[100,100]), ("unknown_input",True),
                                       ("sample_temperature_K",0), ("window",{"type":"tukey","alpha":2})])
def test_invalid_parameters_rejected(key,value):
    p = effective_parameters(defaults())
    p[key] = value
    with pytest.raises(ValueError):
        validate_parameters(p)

def test_ui_and_config_do_not_import_torch():
    # Check package boundary in a fresh process, independent of test imports.
    import subprocess
    result = subprocess.run([sys.executable, "-c",
        "import tacaw_studio.ui,sys; assert 'torch' not in sys.modules; assert 'torched_tacaw' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
