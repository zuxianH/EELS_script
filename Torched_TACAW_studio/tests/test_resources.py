"""Resource sizing against array data and the supplied engine's saved axes."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from tacaw_studio.config import import_config
from tacaw_studio.resources import (estimate_resources, frequency_bins, roi_shape,
                                    measured_run_files, destination_space, GIB)


def test_output_sizes_and_gpu_memory_follow_batch_shape(settings):
    s = copy.deepcopy(settings)
    s["parameters"].update(trajectory_chunks_size=100, frequency_THz_ROI=[-200., 200.])
    s["execution"]["profile"]["backend"] = "cuda"
    one = estimate_resources(s, atom_count=2, element_count=1)
    assert one["shape"] == [1, 100, 5, 1, 8, 8]
    assert one["result_bytes"] == np.empty(one["shape"], dtype=np.float64).nbytes
    assert one["wave_bytes"] == 100 * 8 * 8 * 16
    assert one["gpu_arrays_bytes"] == 3 * one["wave_bytes"]
    assert one["gpu_plan_bytes"] > one["gpu_arrays_bytes"] + GIB
    s["scan"]["batch"] = [5, 1]
    five = estimate_resources(s, atom_count=2, element_count=1)
    assert five["result_bytes"] == one["result_bytes"]
    assert five["wave_bytes"] == 5 * one["wave_bytes"]
    assert five["gpu_arrays_bytes"] == 5 * one["gpu_arrays_bytes"]
    assert five["chunk_bytes"] == 5 * one["chunk_bytes"]


@pytest.mark.parametrize("samples", [4, 7, 8, 600])
@pytest.mark.parametrize("stride", [1, 2, 3])
def test_frequency_count_matches_fft_bins(settings, samples, stride):
    p = settings["parameters"]
    p.update(trajectory_chunks_size=samples, frequency_THz_ROI=[-50., 100.])
    dt = p["trajectory_timestep_fs"] * stride * p["trajectory_chunks_step"] / 1000
    bins = np.fft.fftfreq(samples, dt)
    assert frequency_bins(p, stride) == np.count_nonzero((bins >= -50.) & (bins <= 100.))


def test_cpu_ram_and_slurm_node_estimate(settings):
    s=copy.deepcopy(settings)
    local=estimate_resources(s, atom_count=2, element_count=1)
    assert local["gpu_plan_bytes"] == 0
    assert local["ram_arrays_bytes"] >= local["fft_bytes"]
    s["execution"]["profile"]["scheduler"] = "slurm"
    s["execution"]["slurm"].update(tasks_per_node=4, nodes=2)
    report=estimate_resources(s, atom_count=2, element_count=1)
    assert report["worker_count"] == 8
    assert report["ram_node_plan_bytes"] == local["ram_plan_bytes"] * 4


def test_saved_engine_and_array_dimensions_are_authoritative(settings):
    c={"simulation":{"kspace":{"shape_full":[16,16],"ROI_shape":[4,6]},
                     "frequency_THz":{"ROI_len":3},"n_slices":1},
       "beam":{"scanning":{"shape":[3,2],"batch_shape":[1,2]}},
       "trajectory":{"chunks":{"size":8}}}
    report=estimate_resources(settings, engine_config=c)
    assert report["shape"] == [1,3,3,2,4,6]
    assert report["result_bytes"] == 1*3*3*2*4*6*8
    assert report["prepared"]
    actual=dict(shape=[1,3,3,2,4,6],chunks=[1,3,1,2,4,6],dtype="float64",
                bytes=report["result_bytes"],chunk_bytes=3*2*4*6*8)
    assert estimate_resources(settings, engine_config=c, array_info=actual)["chunk_bytes"] == actual["chunk_bytes"]


def test_provided_example_roi_and_file_sizes_match_saved_engine():
    root=Path(__file__).resolve().parents[3]/"torched_TACAW"
    paths=["SiC/tacaw_results_planewave/scan_gpu_config.yaml",
           "Si_Langevin_300K/tacaw_results/scan_gpu_config.yaml",
           "BTO_Ba-O_bussi/tacaw_results_stemeels_2Dscan/scan_gpu_config.yaml",
           "STO_Ti-O_bussi/tacaw_results_stemeels/scan_gpu_config.yaml"]
    if not root.exists():
        pytest.skip("Provided examples unavailable")
    for relative in paths:
        text=(root/relative).read_text()
        c=yaml.safe_load(text)
        s=import_config(text)
        report=estimate_resources(s, cell_lengths=c["sample"]["unitcell"])
        assert report["roi_shape"] == c["simulation"]["kspace"]["ROI_shape"]
        assert report["frequency_bins"] == c["simulation"]["frequency_THz"]["ROI_len"]
        shape=[1,report["frequency_bins"],*c["beam"]["scanning"]["shape"],*report["roi_shape"]]
        assert report["result_bytes"] == np.prod(shape)*8


def test_conversion_positions_estimate_and_invalid_rois(settings):
    s=copy.deepcopy(settings)
    s["inputs"]["conversion"].update(start=1,stop=11,stride=3)
    report=estimate_resources(s,atom_count=2,source_frames=12)
    assert report["converted_bytes"] == 4*2*3*8
    unknown=estimate_resources(s,atom_count=2)
    assert unknown["converted_bytes"] is None
    p=s["parameters"]
    p.update(kspace_ROI_mode="minmax_mrad",kspace_ROI_min=[-100.,-100.],kspace_ROI_max=[100.,100.])
    with pytest.raises(ValueError,match="Select a trajectory"):
        roi_shape(p)
    p["kspace_ROI_min"],p["kspace_ROI_max"] = [1000.,1000.],[2000.,2000.]
    with pytest.raises(ValueError,match="no grid bins"):
        roi_shape(p,[8,8,8])


def test_measure_files_without_following_external_inputs(tmp_path):
    run=tmp_path/"run"
    run.mkdir()
    (run/"result.zarray").mkdir()
    (run/"result.zarray"/"chunk").write_bytes(b"0"*17)
    (run/"intensity.npy").write_bytes(b"0"*29)
    (run/"scan_coordinates.npy").write_bytes(b"0"*5)
    (run/"trajectories").mkdir()
    (run/"trajectories"/"converted.traj").write_bytes(b"0"*31)
    external=tmp_path/"external.traj"
    external.write_bytes(b"0"*100)
    (run/"trajectories"/"reference.traj").symlink_to(external)
    result=measured_run_files(run)
    assert result["total_bytes"] == 17+29+5+31
    assert result["groups"]["Zarr results"] == 17
    assert result["groups"]["Converted trajectories"] == 31
    assert destination_space(run/"new"/"subfolder")["path"] == str(run)


@pytest.mark.parametrize("backend", ["cpu", "rocm"])
def test_environment_memory_respects_selected_backend(monkeypatch, backend):
    import sys
    from types import SimpleNamespace
    import tacaw_studio.resources as resources
    import tacaw_studio.runner as runner
    allocations=[]
    def properties(index):
        if backend == "cpu":
            raise AssertionError("CPU checks must not require a working GPU driver")
        return SimpleNamespace(name=f"GPU {index}",total_memory=16*GIB)
    torch=SimpleNamespace(version=SimpleNamespace(cuda=None,hip="6.2"),
                          cuda=SimpleNamespace(device_count=lambda:2,is_available=lambda:True,
                                               get_device_properties=properties,synchronize=lambda:None),
                          ones=lambda n,device:allocations.append(device))
    monkeypatch.setitem(sys.modules,"torch",torch)
    monkeypatch.setattr(runner.importlib,"import_module",lambda name:SimpleNamespace(__version__="test",__file__="test"))
    monkeypatch.setattr(resources,"node_memory",lambda:{"total_bytes":64*GIB,"available_bytes":32*GIB})
    info=runner.check_environment(backend)
    assert info["node_memory"]["total_bytes"] == 64*GIB
    if backend == "cpu":
        assert info["gpu_devices"] == [] and not allocations
    else:
        assert len(info["gpu_devices"]) == 2
        assert info["gpu_devices"][0]["total_bytes"] == 16*GIB
        assert allocations == ["cuda:0"]
