from pathlib import Path
import uuid
from streamlit.testing.v1 import AppTest

from tacaw_studio.config import defaults, preset
from tacaw_studio.ui import click_callback

ROOT = Path(__file__).resolve().parents[1]

def find(elements,label):
    return next(e for e in elements if e.label == label)

def submit_config(app, text):
    revision=app.session_state["calc_editor_revision"] if "calc_editor_revision" in app.session_state else 0
    app.session_state[f"calc_editor_{revision}"] = {
        "id": uuid.uuid4().hex, "type": "submit", "text": text,
        "lang": "python", "selected": "", "cursor": ""}
    app.run(timeout=120)


def test_startup_and_parameter_edits(source):
    app = AppTest.from_file(str(ROOT/"app.py"))
    settings = defaults()
    settings["input_folder"] = str(source.parent)
    settings["inputs"]["files"] = [str(source)]
    app.session_state["settings"] = settings
    app.run(timeout=120)
    assert not app.exception
    find(app.selectbox,"Calculation mode").select("2D scan").run()
    assert not app.exception
    # Switching from a horizontal line seeds a valid editable rectangle.
    assert not app.session_state["draft_errors"].get("scan")
    scan = app.session_state["settings"]["scan"]
    assert scan["end"][1] != scan["start"][1]
    assert min(scan["shape"]) >= 2
    find(app.number_input,"End (Å) y").set_value(2.).run()
    find(app.number_input,"Scan points y").set_value(3).run()
    assert not app.exception
    assert app.session_state["settings"]["scan"]["kind"] == "2D scan"
    submit_config(app, app.session_state["calc_text"].replace("beam_energy_keV=60.0", "beam_energy_keV=80.0"))
    assert app.session_state["settings"]["parameters"]["beam_energy_keV"] == 80.
    assert not app.exception

def test_preset_and_python_editor(source):
    app = AppTest.from_file(str(ROOT/"app.py"))
    settings = defaults()
    settings["input_folder"] = str(source.parent)
    settings["inputs"]["files"] = [str(source)]
    app.session_state["settings"] = settings
    app.run(timeout=120)
    find(app.selectbox,"Preset").select("SiC plane wave")
    find(app.button,"Apply preset").click().run()
    assert not app.exception
    assert app.session_state["settings"]["scan"]["kind"] == "Plane wave"
    submit_config(app, app.session_state["calc_text"].replace("beam_energy_keV=60.0", "beam_energy_keV=100.0"))
    assert not app.exception
    assert app.session_state["settings"]["parameters"]["beam_energy_keV"] == 100.
    submit_config(app, "config = Config(unknown_parameter=1)")
    assert not app.exception
    assert any("Unsupported" in e.value for e in app.error)
    assert app.session_state["settings"]["parameters"]["beam_energy_keV"] == 100.
    assert find(app.button,"Generate run folder").disabled


def test_generate_run_from_ui(settings):
    app = AppTest.from_file(str(ROOT/"app.py"))
    app.session_state["settings"] = settings
    app.run(timeout=120)
    find(app.button,"Generate run folder").click().run()
    assert not app.exception
    folder = Path(app.session_state["edit_run_folder"])
    assert (folder/"submit.sh").is_file()
    assert not (folder/"scan_gpu_config.yaml").exists()
    assert find(app.button,"Run / submit")
    assert find(app.metric,"Zarr data (uncompressed)")
    assert find(app.metric,"RAM / worker (approx.)")
    assert find(app.button,"Measure files on disk")


def test_folder_change_replaces_old_selection(source,tmp_path):
    from ase.io.trajectory import Trajectory
    other=tmp_path/"other"
    other.mkdir()
    target=other/"new.traj"
    with Trajectory(source) as inp,Trajectory(target,"w") as out:
        out.write(inp[0])
    app=AppTest.from_file(str(ROOT/"app.py"))
    settings=defaults()
    settings["input_folder"]=str(source.parent)
    settings["inputs"]["files"]=[str(source)]
    app.session_state["settings"]=settings
    app.run(timeout=120)
    find(app.text_input,"Input folder").set_value(str(other)).run()
    assert not app.exception
    assert app.session_state["settings"]["inputs"]["files"] == [str(target)]
    find(app.radio,"Click sets").set_value("End").run()
    assert app.session_state["settings"]["inputs"]["files"] == [str(target)]
    assert not app.exception


def test_rectangle_error_names_the_field_to_change(settings):
    app = AppTest.from_file(str(ROOT / "app.py"))
    app.session_state["settings"] = settings
    app.run(timeout=120)
    find(app.selectbox, "Calculation mode").select("2D scan").run()
    find(app.number_input, "End (Å) y").set_value(1.).run()
    find(app.number_input, "Start / origin (Å) y").set_value(1.).run()
    assert not app.exception
    assert any("End y equals Start y" in e.value for e in app.error)
    find(app.number_input, "End (Å) y").set_value(2.).run()
    assert not app.session_state["draft_errors"].get("scan")


def test_code_draft_survives_scan_changes_until_reload(settings):
    app=AppTest.from_file(str(ROOT / "app.py"))
    app.session_state["settings"]=settings
    app.run(timeout=120)
    draft=app.session_state["calc_text"].replace("beam_energy_keV=60.0", "beam_energy_keV=90.0")
    revision=app.session_state["calc_editor_revision"] if "calc_editor_revision" in app.session_state else 0
    app.session_state[f"calc_editor_{revision}"]={"id":uuid.uuid4().hex,"type":"blur","text":draft,
                                               "lang":"python","selected":"","cursor":""}
    app.run(timeout=120)
    assert app.session_state["settings"]["parameters"]["beam_energy_keV"] == 60.
    assert app.session_state["calc_text"] == draft
    assert find(app.button,"Generate run folder").disabled
    find(app.selectbox,"Calculation mode").select("2D scan").run()
    assert not app.exception
    assert app.session_state["calc_text"] == draft
    assert app.session_state["calc_external_change"]
    find(app.button,"Reload current settings").click().run()
    assert not app.exception
    assert "scanning_shape=[5, 2]" in app.session_state["calc_text"]
    assert "beam_energy_keV=60.0" in app.session_state["calc_text"]
    assert "calculation_editor" not in app.session_state["draft_errors"]
