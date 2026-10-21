"""Real arbitrary-coordinate click/overlay checks (optional installed browser)."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

import pytest

playwright = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def browser_page(source):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port = sock.getsockname()[1]
    process = subprocess.Popen([sys.executable,"-m","streamlit","run",str(ROOT/"app.py"),
                               "--server.address=127.0.0.1",f"--server.port={port}",
                               "--server.fileWatcherType=none","--browser.gatherUsageStats=false"],
                               cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(200):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=.5).close()
                break
            except OSError:
                if process.poll() is not None:
                    pytest.fail("Streamlit process exited")
                time.sleep(.1)
        else:
            pytest.fail("Streamlit did not become healthy")
        with playwright.sync_playwright() as driver:
            executable = os.environ.get("TACAW_BROWSER_PATH")
            try:
                engine = os.environ.get("TACAW_BROWSER", "chromium")
                browser = getattr(driver, engine).launch(headless=True, timeout=30000,
                            **({"executable_path":executable} if executable else {}))
            except playwright.Error as exc:
                pytest.skip(f"Browser unavailable: {exc}")
            page = browser.new_page(viewport={"width":1550,"height":1200})
            errors = []
            page.on("pageerror",lambda err: errors.append(str(err)))
            page.set_default_timeout(30000)
            page.goto(f"http://127.0.0.1:{port}")
            field = page.get_by_role("textbox",name="Input folder",exact=True)
            field.fill(str(source.parent))
            field.press("Enter")
            page.get_by_role("tab",name="Scan",exact=True).click()
            page.locator(".st-key-trajectory_plot .js-plotly-plot").wait_for()
            yield page
            assert not page.locator('[data-testid="stException"]').count()
            assert not errors
            browser.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def click_xy(page,x,y):
    chart=page.locator(".st-key-trajectory_plot .js-plotly-plot")
    page.wait_for_function("""() => {
        const c=document.querySelector('.st-key-trajectory_plot .js-plotly-plot');
        return c?._fullLayout?.xaxis && c?._fullLayout?.yaxis;
    }""")
    for attempt in range(5):
        try:
            chart.scroll_into_view_if_needed()
            break
        except playwright.Error:
            if attempt == 4:
                raise
            page.wait_for_timeout(200)
    pixel=chart.evaluate("""(c,p) => {
        const box=c.getBoundingClientRect(),l=c._fullLayout;
        return {x:box.x+l.xaxis._offset+l.xaxis.l2p(p[0]),
                y:box.y+l.yaxis._offset+l.yaxis.l2p(p[1])};
    }""",[x,y])
    page.mouse.click(pixel["x"],pixel["y"])


def test_arbitrary_click_updates_endpoint_and_overlay(browser_page):
    page=browser_page
    chart=page.locator(".st-key-trajectory_plot .js-plotly-plot")
    preview_panel=page.get_by_test_id("stExpander").filter(
        has=page.get_by_text("Trajectory preview settings",exact=True))
    assert preview_panel.locator("details").get_attribute("open") is None
    assert chart.bounding_box()["y"] < preview_panel.bounding_box()["y"]
    scan_panel=page.get_by_test_id("stExpander").filter(
        has=page.get_by_text("Scan settings",exact=True))
    assert scan_panel.locator("details").get_attribute("open") is not None
    assert scan_panel.bounding_box()["x"] + scan_panel.bounding_box()["width"] <= chart.bounding_box()["x"]
    # The scan controls are visible on the left, beside the plot on the right.
    page.get_by_text("End",exact=True).click()
    page.wait_for_timeout(1000)
    page.get_by_text("Selection target: End.",exact=False).wait_for()
    page.wait_for_timeout(500)
    click_xy(page,2.2,1.7)
    end_x=page.get_by_role("spinbutton",name="End (Å) x",exact=True)
    end_y=page.get_by_role("spinbutton",name="End (Å) y",exact=True)
    page.wait_for_function("""() => {
        const x=document.querySelector('.st-key-edit_scan_end_0 input'),
              y=document.querySelector('.st-key-edit_scan_end_1 input');
        return x && y && Math.abs(Number(x.value)-2.2)<.02 && Math.abs(Number(y.value)-1.7)<.02;
    }""")
    assert float(end_x.input_value()) == pytest.approx(2.2,abs=.02)
    assert float(end_y.input_value()) == pytest.approx(1.7,abs=.02)
    chart=page.locator(".st-key-trajectory_plot .js-plotly-plot")
    page.wait_for_function("""() => {
        const c=document.querySelector('.st-key-trajectory_plot .js-plotly-plot');
        const scan=c?.data?.find(t=>t.name==='Probe positions');
        return scan && Math.abs(scan.x.at(-1)-2.2)<.02 && Math.abs(scan.y.at(-1)-1.7)<.02;
    }""")
    page.get_by_text("Snap selected coordinates to nearest displayed atom",exact=True).click()
    page.wait_for_timeout(500)
    click_xy(page,2.95,3.05)
    page.wait_for_function("""() => {
        const x=document.querySelector('.st-key-edit_scan_end_0 input'),
              y=document.querySelector('.st-key-edit_scan_end_1 input');
        return x && y && Math.abs(Number(x.value)-3)<1e-6 && Math.abs(Number(y.value)-3)<1e-6;
    }""")
    # Switch to a rectangular scan and verify its far corner and ij probe count.
    page.get_by_role("combobox",name="Calculation mode",exact=True).click()
    page.get_by_role("option",name="2D scan",exact=True).click()
    y_points=page.get_by_role("spinbutton",name="Scan points y",exact=True)
    y_points.fill("3")
    y_points.press("Enter")
    page.wait_for_function("""() => {
        const c=document.querySelector('.st-key-trajectory_plot .js-plotly-plot');
        const scan=c?.data?.find(t=>t.name==='Probe positions');
        return scan && scan.x.length===15 && Math.abs(scan.x.at(-1)-3)<1e-6 && Math.abs(scan.y.at(-1)-3)<1e-6;
    }""")
    page.screenshot(path=os.environ.get("TACAW_SCREENSHOT","/tmp/tacaw-studio-browser.png"),full_page=True)


def test_highlighted_python_configuration_editor(browser_page):
    page=browser_page
    page.get_by_role("combobox",name="Calculation mode",exact=True).click()
    page.get_by_role("option",name="2D scan",exact=True).click()
    page.get_by_role("spinbutton",name="Scan points y",exact=True).wait_for()
    page.get_by_role("tab",name="Calculation",exact=True).click()
    frame=page.frame_locator('iframe[src*="code_editor"]')
    editor=frame.locator(".ace_editor")
    editor.wait_for()
    frame.locator(".ace_line").filter(has_text="scanning_shape=[5, 2]").wait_for()
    assert frame.locator(".ace_constant.ace_numeric").count() > 0
    assert frame.locator(".ace_string").count() > 0
    original=editor.evaluate("el => el.env.editor.getValue()")
    modified=original.replace("beam_energy_keV=60.0", "beam_energy_keV=80.0")
    assert modified != original
    editor.evaluate("(el, text) => el.env.editor.setValue(text, -1)", modified)
    frame.get_by_role("button",name="Apply configuration",exact=True).click()
    page.get_by_text("Configuration applied. Scan, inputs and resource estimates updated.",exact=True).wait_for()
    with page.expect_download() as downloaded:
        page.get_by_role("button",name="Download applied TACAW YAML",exact=True).click()
    import yaml
    applied=yaml.safe_load(Path(downloaded.value.path()).read_text())
    assert applied["beam_energy_keV"] == 80.
    assert applied["scanning_shape"] == [5,2]
    page.get_by_role("tab",name="Scan",exact=True).click()
    # Endpoint and two-axis controls distinguish the rectangle from raw basis mode.
    end_y=page.get_by_role("spinbutton",name="End (Å) y",exact=True)
    end_y.wait_for()
    assert float(end_y.input_value()) == 1.
    assert int(page.get_by_role("spinbutton",name="Scan points y",exact=True).input_value()) == 2
    page.get_by_role("tab",name="Calculation",exact=True).click()
    editor=frame.locator(".ace_editor")
    editor.wait_for()
    editor.evaluate("el => el.env.editor.setValue('config = Config(unknown_parameter=1)', -1)")
    frame.get_by_role("button",name="Apply configuration",exact=True).click()
    page.get_by_text("Unsupported or runtime-owned parameter: unknown_parameter",exact=True).wait_for()
    page.get_by_role("button",name="Reload current settings",exact=True).click()
    page.wait_for_function("""() => !Array.from(document.querySelectorAll('[data-testid="stAlert"]')).some(
        e => e.textContent.includes('unknown_parameter'))""")
    frame.locator(".ace_editor").wait_for()
    frame.locator(".ace_line").filter(has_text="beam_energy_keV=80.0").wait_for()
    page.wait_for_function("""() => {
        const frame = document.querySelector('iframe[src*="code_editor"]');
        return frame && frame.getBoundingClientRect().height > 600;
    }""")
    page.locator('[data-testid="stMain"]').evaluate("el => el.scrollTo(0, 0)")
    page.screenshot(path="/tmp/tacaw-python-editor.png",full_page=True)
