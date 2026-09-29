"""Real mouse and keyboard coverage for the resizable spectrum workspace."""
import json

import pytest
from panel_layout import ADJUSTABLE_PANELS_ENABLED

pytestmark = pytest.mark.skipif(
    not ADJUSTABLE_PANELS_ENABLED, reason="Adjustable panels are temporarily disabled")

from test_axis_browser import spectrum_page


def wait_ready(page):
    page.get_by_role('separator', name='Resize Detector and EELS Spectrum panels').wait_for(state='visible')
    page.get_by_role('separator', name='Resize EELS Spectrum height', exact=True).wait_for(state='visible')


def drag(page, locator, dx=0, dy=0):
    locator.hover()
    revision = page.locator(".eels-height-handle").first.get_attribute("data-render-revision")
    box = locator.bounding_box()
    x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + dx, y + dy, steps=15)
    page.mouse.up()
    page.wait_for_function("""revision => Number(document.querySelector('.eels-height-handle')?.dataset.renderRevision) > Number(revision)""", arg=revision)


def test_resize_persist_config_and_reset(spectrum_page, tmp_path):
    page = spectrum_page
    wait_ready(page)
    detector = page.locator('.st-key-detector_panel')
    initial_width = detector.bounding_box()['width']
    drag(page, page.get_by_role('separator', name='Resize Detector and EELS Spectrum panels'), dx=100)
    page.wait_for_function("""width => document.querySelector('.st-key-detector_panel').getBoundingClientRect().width > width + 70""", arg=initial_width)
    drag(page, page.get_by_role('separator', name='Resize EELS Spectrum height', exact=True), dy=90)
    page.wait_for_function("""() => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout.height >= 520""")
    drag(page, page.get_by_role('separator', name='Resize Detector height', exact=True), dy=60)
    page.wait_for_function("""() => document.querySelector('.st-key-detector_click_target .js-plotly-plot')._fullLayout.height >= 470""")
    # Changing extraction parameters must retain the layout saved by the gestures.
    radius = page.get_by_role('spinbutton', name='radius', exact=True)
    radius.fill('10')
    radius.press('Enter')
    page.wait_for_function("""() => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout.height >= 520""")
    assert detector.bounding_box()['width'] > initial_width + 70
    with page.expect_download() as downloaded:
        page.get_by_role('button', name='Save config', exact=True).click()
    path = tmp_path / 'layout.json'
    downloaded.value.save_as(path)
    settings = json.loads(path.read_text())['settings']
    assert settings['panel_detector_width'] > 1 / 3
    assert settings['panel_spectrum_height'] >= 520
    assert settings['panel_detector_height'] >= 470
    page.get_by_role('button', name='Reset panel layout').click()
    page.wait_for_function("""() => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout.height === 520""")
    assert abs(detector.bounding_box()['width'] - initial_width) < 4
    page.locator('input[type=file]').set_input_files(path)
    page.get_by_role('button', name='Load config', exact=True).click()
    page.wait_for_function("""() => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout.height >= 520""")
    assert detector.bounding_box()['width'] > initial_width + 70
    page.screenshot(path=str(tmp_path / 'resized-panels.png'), full_page=True)


def test_move_panels_hide_detector_and_narrow_layout(spectrum_page):
    page = spectrum_page
    wait_ready(page)
    detector = page.locator('.st-key-detector_panel')
    spectrum = page.locator('.st-key-spectrum_panel')
    assert detector.bounding_box()['x'] < spectrum.bounding_box()['x']
    revision = page.locator('.eels-height-handle').first.get_attribute('data-render-revision')
    detector.locator('.eels-panel-title').drag_to(spectrum.locator('.eels-panel-title'))
    page.wait_for_function("""revision => Number(document.querySelector('.eels-height-handle')?.dataset.renderRevision) > Number(revision)""", arg=revision)
    page.wait_for_function("""() => document.querySelector('.st-key-detector_panel').getBoundingClientRect().x > document.querySelector('.st-key-spectrum_panel').getBoundingClientRect().x""")
    # Keyboard resizing remains usable after swapping.
    separator = page.get_by_role('separator', name='Resize EELS Spectrum height', exact=True)
    separator.focus()
    page.keyboard.press('ArrowDown')
    page.wait_for_function("""() => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout.height === 464""")
    page.get_by_label('Show detector preview', exact=True).locator('xpath=ancestor::label').click()
    page.locator('.st-key-detector_panel').wait_for(state='detached')
    page.get_by_role('separator', name='Resize EELS Spectrum height', exact=True).wait_for(state='visible')
    assert page.get_by_role('separator', name='Resize Detector and EELS Spectrum panels').count() == 0
    page.get_by_label('Show detector preview', exact=True).locator('xpath=ancestor::label').click()
    wait_ready(page)
    page.set_viewport_size({'width': 800, 'height': 1100})
    page.wait_for_function("""() => {
        const a = document.querySelector('.st-key-detector_panel').getBoundingClientRect();
        const b = document.querySelector('.st-key-spectrum_panel').getBoundingClientRect();
        return Math.abs(a.x - b.x) < 2 && a.y >= b.bottom;
    }""")
    assert not page.get_by_role('separator', name='Resize Detector and EELS Spectrum panels').is_visible()
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
