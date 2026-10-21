"""Versioned Studio settings and a flat-input adapter for TACAW Config."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import numpy as np
import yaml

from .geometry import scan_parameters, scan_from_parameters

# Inputs accepted by the local 0.3.1 engine. Runtime-owned path/status fields
# are kept outside the user's flat parameter editor.
DEFAULT_PARAMETERS = {
    "name": "TACAW run", "info": "", "beam_energy_keV": 60.,
    "beam_conv_ang_mrad": 20., "sample_name": "sample",
    "sample_temperature_K": 300., "sample_supercell_int": [1, 1, 1],
    "trajectory_timestep_fs": 2.5, "trajectory_chunks_size": 600,
    "trajectory_chunks_skip_init": 500, "trajectory_chunks_step": 3,
    "trajectory_chunks_nof": 3, "trajectory_chunks_overlap": 2.,
    "kspace_shape_full": [400, 400], "kspace_ROI_mode": "minmax_mrad",
    "kspace_ROI_shape": None, "kspace_ROI_min": [-100., -100.],
    "kspace_ROI_max": [100., 100.], "kspace_bandwidth_limiting": [2/3, 2/3],
    "n_slices": 208, "frequency_THz_ROI": [-100., 100.], "window": "hann",
    "device": "cpu", "center_atoms_in_cell": False, "comments": [],
    "scanning_mode": "parallelogram", "scanning_shape": [5, 1],
    "scanning_batch_shape": [1, 1], "scanning_origin": [0., 0.],
    "scanning_basis_vectors": [[1.25, 0.], [0., 1.]],
}
OPTIONAL_PARAMETERS = {"sample_structure_file", "scanning_box_end", "scanning_box_size"}
MANAGED_PARAMETERS = {"datafolder", "config_file", "trajectory_file", "trajectory_files",
                      "trajectory_file_type", "trajectory_file_kwargs", "trajectory_len",
                      "intensities_zarray", "dump_to_yaml",
                      "local_project_directory", "remote_project_directory"}
SCAN_KEYS = {"scanning_mode", "scanning_shape", "scanning_batch_shape",
             "scanning_origin", "scanning_basis_vectors", "scanning_box_end",
             "scanning_box_size"}
SCHEMA_VERSION = 1


def defaults() -> dict:
    from .profiles import default_profile, default_slurm
    return {
        "schema_version": SCHEMA_VERSION, "input_folder": str(Path.cwd()),
        "inputs": {"files": [], "recursive": False, "bead_set": False,
                   "conversion": {"format": "auto", "start": 0, "stop": None,
                                  "stride": 1, "positions_unit": "auto",
                                  "cell_unit": "auto", "cell": None, "pbc": None,
                                  "type_symbols": {}}},
        "parameters": copy.deepcopy(DEFAULT_PARAMETERS),
        "scan": {"kind": "Line scan", "shape": [5, 1], "batch": [1, 1],
                 "start": [0., 0.], "end": [1., 0.],
                 "basis": [[1., 0.], [0., 1.]], "periodic": False},
        "execution": {"profile": default_profile(), "slurm": default_slurm()},
        "auto_npy": False, "run_name": "tacaw", "output_root": "",
        "intensities_name": "scan_gpu_tacaw.zarray",
    }


def effective_parameters(settings: dict) -> dict:
    p = copy.deepcopy(settings["parameters"])
    if settings["scan"]["kind"] != "Native box":
        for key in SCAN_KEYS:
            p.pop(key, None)
        p.update(scan_parameters(settings["scan"]))
    return p


def apply_parameters(settings: dict, parameters: dict) -> dict:
    validate_parameters(parameters)
    result = copy.deepcopy(settings)
    result["parameters"] = copy.deepcopy(parameters)
    result["scan"] = scan_from_parameters(parameters)
    return result


def vector(p: dict, key: str, n: int, *, integer=False, positive=False):
    value = p.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{key} must have {n} values.")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or
           not math.isfinite(x) or (integer and not isinstance(x, int)) or
           (positive and x <= 0) for x in value):
        raise ValueError(f"{key} contains invalid values.")


def validate_parameters(p: dict) -> None:
    if not isinstance(p, dict):
        raise ValueError("Parameters must be a YAML mapping.")
    unknown = set(p) - set(DEFAULT_PARAMETERS) - OPTIONAL_PARAMETERS
    if unknown:
        raise ValueError(f"Unsupported or runtime-owned parameters: {', '.join(sorted(unknown))}")
    for key in ("beam_energy_keV", "sample_temperature_K", "trajectory_timestep_fs",
                "trajectory_chunks_overlap"):
        value = p.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a finite positive number.")
    conv = p.get("beam_conv_ang_mrad")
    if not isinstance(conv, (int, float)) or not math.isfinite(conv) or conv < 0:
        raise ValueError("Convergence angle must be finite and nonnegative.")
    for key in ("trajectory_chunks_size", "trajectory_chunks_step", "n_slices"):
        if type(p.get(key)) is not int or p[key] < (2 if key == "trajectory_chunks_size" else 1):
            raise ValueError(f"{key} must be a positive integer (chunk size at least 2).")
    if type(p.get("trajectory_chunks_skip_init")) is not int or p["trajectory_chunks_skip_init"] < 0:
        raise ValueError("Initial skipped frames must be a nonnegative integer.")
    if p.get("trajectory_chunks_nof") is not None and (
            type(p["trajectory_chunks_nof"]) is not int or p["trajectory_chunks_nof"] < 1):
        raise ValueError("Chunk count must be null (all available) or a positive integer.")
    if int(p["trajectory_chunks_size"] * p["trajectory_chunks_step"] / p["trajectory_chunks_overlap"]) < 1:
        raise ValueError("Overlap makes the chunk-start increment zero.")
    vector(p, "sample_supercell_int", 3, integer=True, positive=True)
    for key in ("scanning_shape", "scanning_batch_shape", "kspace_shape_full"):
        vector(p, key, 2, integer=True, positive=True)
    if any(n % b for n, b in zip(p["scanning_shape"], p["scanning_batch_shape"])):
        raise ValueError("Scan batch shape must divide scan shape.")
    vector(p, "kspace_bandwidth_limiting", 2, positive=True)
    if max(p["kspace_bandwidth_limiting"]) > 1:
        raise ValueError("Bandwidth limits must be in (0, 1].")
    vector(p, "frequency_THz_ROI", 2)
    if p["frequency_THz_ROI"][0] >= p["frequency_THz_ROI"][1]:
        raise ValueError("Frequency ROI minimum must be below maximum.")
    mode = p.get("kspace_ROI_mode")
    if mode not in (None, "center", "minmax_mrad", "mrad", "minmaxmrad"):
        raise ValueError("Invalid k-space ROI mode.")
    if mode == "center" and p.get("kspace_ROI_shape") is not None:
        vector(p, "kspace_ROI_shape", 2, integer=True, positive=True)
        limits = (np.array(p["kspace_shape_full"]) * p["kspace_bandwidth_limiting"]).astype(int)
        if np.any(np.array(p["kspace_ROI_shape"]) > limits):
            raise ValueError("Central ROI exceeds the bandwidth-limited grid.")
    if mode in ("minmax_mrad", "mrad", "minmaxmrad"):
        vector(p, "kspace_ROI_min", 2)
        vector(p, "kspace_ROI_max", 2)
        if any(a >= b for a, b in zip(p["kspace_ROI_min"], p["kspace_ROI_max"])):
            raise ValueError("K-space ROI minima must be below maxima.")
    if p.get("scanning_mode") not in ("unitcell", "box", "box_relative", "parallelogram"):
        raise ValueError("Invalid scanning mode.")
    if p["scanning_mode"] != "unitcell":
        vector(p, "scanning_origin", 2)
    if p["scanning_mode"] == "parallelogram":
        scan_parameters(scan_from_parameters(p))
    if p["scanning_mode"] in ("box", "box_relative"):
        key = "scanning_box_size" if p.get("scanning_box_size") is not None else "scanning_box_end"
        vector(p, key, 2)
        if key == "scanning_box_size" and any(x == 0 for x in p[key]):
            raise ValueError("Box dimensions cannot be zero.")
        if key == "scanning_box_end" and any(a == b for a, b in zip(p[key], p["scanning_origin"])):
            raise ValueError("Box width and height cannot be zero.")
    w = p.get("window")
    if w != "hann":
        if not isinstance(w, dict) or w.get("type") not in ("hann", "tukey"):
            raise ValueError("Window must be hann or a hann/tukey mapping.")
        if w["type"] == "tukey" and not 0 <= w.get("alpha", -1) <= 1:
            raise ValueError("Tukey alpha must be between 0 and 1.")
    if not isinstance(p.get("device"), str) or not p["device"].strip():
        raise ValueError("Device must be a Torch device string.")
    if type(p.get("center_atoms_in_cell")) is not bool:
        raise ValueError("center_atoms_in_cell must be boolean.")
    for key in ("name", "info", "sample_name"):
        if not isinstance(p.get(key), str):
            raise ValueError(f"{key} must be text.")


def validate_settings(s: dict, *, require_files=False) -> None:
    if s.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Studio schema version.")
    validate_parameters(effective_parameters(s))
    inputs = s["inputs"]
    files = inputs["files"]
    if not isinstance(files, list) or any(not isinstance(f, str) for f in files):
        raise ValueError("Trajectory files must be a list of paths.")
    if require_files and not files:
        raise ValueError("Select at least one trajectory.")
    if len(files) > 1 and not inputs.get("bead_set"):
        raise ValueError("Enable synchronized bead set to select more than one trajectory.")
    if len(set(str(Path(f).expanduser().resolve()) for f in files)) != len(files):
        raise ValueError("The same trajectory cannot be selected twice.")
    c = inputs["conversion"]
    if type(c["start"]) is not int or c["start"] < 0 or type(c["stride"]) is not int or c["stride"] < 1:
        raise ValueError("Conversion start must be nonnegative and stride positive.")
    if c["stop"] is not None and (type(c["stop"]) is not int or c["stop"] <= c["start"]):
        raise ValueError("Conversion stop must be greater than start.")
    if c["format"] not in ("auto", "xyz", "extxyz", "ipi", "lammps-dump-text"):
        raise ValueError("Unsupported conversion format.")
    for key in ("positions_unit", "cell_unit"):
        if c[key] not in ("auto", "angstrom", "bohr"):
            raise ValueError(f"Invalid {key}.")
    if c["cell"] is not None:
        value = np.asarray(c["cell"], float)
        if value.shape not in ((3,), (6,), (3, 3)) or not np.isfinite(value).all():
            raise ValueError("Cell override needs 3 lengths, 6 cell parameters, or a 3x3 matrix.")
    if c["pbc"] is not None and (
            not isinstance(c["pbc"], list) or len(c["pbc"]) != 3 or any(type(x) is not bool for x in c["pbc"])):
        raise ValueError("PBC override must be [true/false, true/false, true/false].")
    name = s.get("intensities_name", "")
    if Path(name).name != name or not name.endswith((".zarray", ".zarr")):
        raise ValueError("Intensity filename must be a basename ending in .zarray or .zarr.")
    from .profiles import validate_profile, validate_slurm
    validate_profile(s["execution"]["profile"])
    validate_slurm(s["execution"]["slurm"], s["execution"]["profile"]["backend"])


def dump_settings(s: dict) -> str:
    return yaml.safe_dump(s, sort_keys=False)


def import_config(text: str, current: dict | None = None) -> dict:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    if "schema_version" in data:
        validate_settings(data)
        return copy.deepcopy(data)
    result = copy.deepcopy(current or defaults())
    p = copy.deepcopy(DEFAULT_PARAMETERS)
    if "beam" in data:
        beam, sample = data["beam"], data["sample"]
        for k in ("energy_keV", "conv_ang_mrad"):
            p["beam_" + k] = beam[k]
        for k in ("mode", "shape", "batch_shape", "origin", "basis_vectors", "box_size", "box_end"):
            if k in beam["scanning"]:
                p["scanning_" + k] = beam["scanning"][k]
        for k in ("name", "supercell_int", "temperature_K", "structure_file"):
            if k in sample:
                p["sample_" + k] = sample[k]
        traj, sim = data["trajectory"], data["simulation"]
        p["trajectory_timestep_fs"] = traj["timestep_fs"]
        for k in ("size", "skip_init", "step", "nof", "overlap"):
            p["trajectory_chunks_" + k] = traj["chunks"].get(k, p["trajectory_chunks_" + k])
        for k in ("shape_full", "ROI_mode", "ROI_shape", "ROI_min", "ROI_max", "bandwidth_limiting"):
            if k in sim["kspace"]:
                p["kspace_" + k] = sim["kspace"][k]
        for k in ("n_slices", "window", "device", "center_atoms_in_cell"):
            p[k] = sim.get(k, p[k])
        p["frequency_THz_ROI"] = sim["frequency_THz"]["ROI"]
        for k in ("name", "info", "comments"):
            p[k] = data.get(k, p[k])
        files = traj.get("files", [traj.get("file")])
    else:
        files = data.get("trajectory_files", [data.get("trajectory_file")])
        p.update({k: v for k, v in data.items() if k not in MANAGED_PARAMETERS})
    # Do not retain incompatible stale geometric options from defaults.
    if p["scanning_mode"] != "parallelogram":
        p.pop("scanning_basis_vectors", None)
    if p["scanning_mode"] == "unitcell":
        p.pop("scanning_origin", None)
    result = apply_parameters(result, p)
    if files and all(files):
        result["inputs"]["files"] = [str(f) for f in files]
        result["inputs"]["bead_set"] = len(files) > 1
        result["input_folder"] = str(Path(files[0]).parent)
    return result


def preset(name: str, current: dict) -> dict:
    s = copy.deepcopy(current)
    p = copy.deepcopy(DEFAULT_PARAMETERS)
    p["name"], p["sample_name"] = name, name.split()[0]
    s["run_name"] = name.replace(" ", "_")
    if name == "SiC plane wave":
        p.update(beam_conv_ang_mrad=0., kspace_shape_full=[800, 800], trajectory_chunks_nof=2)
        s["scan"] = dict(kind="Plane wave", shape=[1, 1], batch=[1, 1],
                         start=[0., 0.], end=[1., 0.], basis=[[1., 0.], [0., 1.]], periodic=False)
    elif name == "Si line scan":
        p.update(kspace_shape_full=[400, 399], trajectory_chunks_overlap=1.)
        start, basis = [12.9273, 14.4315], [[6.96221794872/4, 0.], [0., 1.]]
        s["scan"] = dict(kind="Line scan", shape=[5, 1], batch=[5, 1], start=start,
                         end=(np.array(start) + np.array(basis[0])*4/5).tolist(),
                         basis=basis, periodic=False)
    elif name == "BTO 2D scan":
        p.update(sample_name="BaTiO3", beam_conv_ang_mrad=33., trajectory_timestep_fs=5.,
                 trajectory_chunks_skip_init=1000, trajectory_chunks_nof=30,
                 kspace_shape_full=[800, 800], n_slices=200,
                 kspace_ROI_min=[-120., -120.], kspace_ROI_max=[120., 120.])
        start = [10.853339269 - 4.023304441*.25, 20.853681137 - 4.014531257*.25]
        end = (np.array(start) + np.array([4.023304441, 4.014531257])*1.5).tolist()
        s["scan"] = dict(kind="2D scan", shape=[30, 30], batch=[3, 3],
                         start=start, end=end, basis=[[1., 0.], [0., 1.]], periodic=True)
    elif name == "STO line scan":
        p.update(sample_name="SrTiO3", beam_conv_ang_mrad=33., trajectory_timestep_fs=5.,
                 trajectory_chunks_skip_init=1000, trajectory_chunks_nof=60,
                 kspace_shape_full=[800, 800], n_slices=200,
                 kspace_ROI_min=[-120., -120.], kspace_ROI_max=[120., 120.])
        start, basis = [20.524052-2.0045128205125*4, 22.471618], [[16.03610256410, 0.], [0., 1.]]
        s["scan"] = dict(kind="Line scan", shape=[40, 1], batch=[20, 1], start=start,
                         end=(np.array(start)+np.array(basis[0])*39/40).tolist(),
                         basis=basis, periodic=False)
    else:
        raise ValueError("Unknown preset.")
    s["parameters"] = p
    s["parameters"] = effective_parameters(s)
    return s
