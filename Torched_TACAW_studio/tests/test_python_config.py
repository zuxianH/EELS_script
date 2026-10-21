import copy
import sys
from pathlib import Path

import pytest

from tacaw_studio.config import effective_parameters, preset
from tacaw_studio.python_config import apply_python_config, render_python_config, parse_python_config


def test_roundtrip_preserves_line_geometry_and_values(settings):
    before=copy.deepcopy(settings)
    code=render_python_config(settings)
    after=apply_python_config(settings,code)
    assert after["scan"] == before["scan"]
    assert effective_parameters(after) == effective_parameters(before)
    assert settings == before
    updated=apply_python_config(settings,code.replace("beam_energy_keV=60.0","beam_energy_keV=80.0"))
    assert updated["parameters"]["beam_energy_keV"] == 80.
    assert updated["scan"] == before["scan"]


def test_variables_arithmetic_strings_and_path_mapping(settings,tmp_path):
    code=f'''from torched_tacaw import Config
job_tag = "test"
temperature = 300 + 20
out_dir = {str(tmp_path / "runs")!r}
traj_file = {settings["inputs"]["files"][0]!r}
config = Config(
    name=f"Si 110 {{job_tag}}",
    datafolder=str(out_dir) + "/",
    trajectory_file=str(traj_file),
    sample_temperature_K=temperature,
    kspace_bandwidth_limiting=[2/3, 2/3],
    intensities_zarray="result.zarr",
)
'''
    result=apply_python_config(settings,code)
    assert result["parameters"]["name"] == "Si 110 test"
    assert result["parameters"]["sample_temperature_K"] == 320
    assert result["output_root"] == str(tmp_path / "runs")
    assert result["inputs"]["files"] == settings["inputs"]["files"]
    assert result["intensities_name"] == "result.zarr"
    assert result["scan"] == settings["scan"]


@pytest.mark.parametrize("name",["SiC plane wave","Si line scan","BTO 2D scan","STO line scan"])
def test_presets_roundtrip_through_python_editor(settings,name):
    initial=preset(name,settings)
    result=apply_python_config(initial,render_python_config(initial))
    assert result["scan"] == initial["scan"]
    assert effective_parameters(result) == effective_parameters(initial)


def test_geometry_changes_update_scan(settings):
    result=apply_python_config(settings,'config = Config(scanning_mode="unitcell", scanning_shape=[4,4])')
    assert result["scan"]["kind"] == "Unit cell"
    assert effective_parameters(result)["scanning_shape"] == [4,4]


def test_python_editor_can_recover_invalid_rectangle(settings):
    invalid=copy.deepcopy(settings)
    invalid["scan"].update(kind="2D scan",end=[1,0],shape=[5,2])
    code=render_python_config(invalid)
    result=apply_python_config(invalid,code)
    assert effective_parameters(result)
    assert result["scan"]["kind"] == "Basis vectors"


@pytest.mark.parametrize("code, message",[
    ('config=Config(beam_energy_keV=args.energy)', "Replace args.*"),
    ('config=Config(beam_energy_keV=-1)', "positive"),
    ('config=Config(beam_energy_keV=)', "Line 1"),
    ('config=Config(unknown=1)', "Unsupported"),
    ('config=Config(config_file="existing.yaml")', "generated"),
    ('config=Config(trajectory_file="a",trajectory_files=["b"])', "not both"),
    ('config=Config(beam_energy_keV=1/0)', "Invalid Python"),
    ('config=Config(sample_structure_file=123)', "path string"),
    ('config=Config(beam_energy_keV=60,beam_energy_keV=80)', "twice"),
    ('config=Config(beam_energy_keV=2**1000000)', "powers"),
    ('config=Config(); config=Config()', "exactly one"),
])
def test_bad_config_leaves_settings_unchanged(settings,code,message):
    original=copy.deepcopy(settings)
    with pytest.raises(ValueError,match=message):
        apply_python_config(settings,code)
    assert settings == original


def test_parser_never_executes_calls_or_imports(settings,tmp_path):
    target=tmp_path/"do_not_create"
    code=f'config=Config(info=__import__("pathlib").Path({str(target)!r}).touch())'
    with pytest.raises(ValueError):
        apply_python_config(settings,code)
    assert not target.exists()
    with pytest.raises(ValueError):
        parse_python_config('import os\nconfig=Config()')


def test_focused_probe_can_be_selected_from_plane_wave(settings):
    plane=preset("SiC plane wave",settings)
    changed=apply_python_config(plane,"config = Config(beam_conv_ang_mrad=20.0)")
    assert changed["scan"]["kind"] == "Unit cell"
    assert effective_parameters(changed)["beam_conv_ang_mrad"] == 20.
