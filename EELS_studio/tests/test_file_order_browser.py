"""Drag comparison files and verify the rendered spectra and saved workspace."""
import json

import numpy as np

from test_axis_browser import spectrum_page


def test_drag_files_updates_plot_and_saved_order(spectrum_page, tmp_path):
    page = spectrum_page
    for name, value in [('scan_T600K.npy', 2.), ('scan_T900K.npy', 3.)]:
        np.save(tmp_path / name, np.full((41, 12, 14), value))
    page.get_by_role('button', name='Refresh files', exact=True).click()
    for name in ['scan_T600K.npy', 'scan_T900K.npy']:
        page.get_by_role('combobox', name='Files to compare', exact=True).click()
        page.get_by_role('option', name=name, exact=True).click()
        page.keyboard.press('Escape')
    rows = page.locator('[data-testid="file-order-list"] li')
    page.wait_for_function("""() => {
        const host = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return host?.data?.length === 3;
    }""")
    rows.last.wait_for()
    assert rows.count() == 3
    chart = page.locator('.st-key-spectrum_axis_target .js-plotly-plot')
    original = chart.evaluate('c => c.data.map(t => ({name:t.name, line:t.line}))')
    first_path = str(tmp_path / 'scan_T300K.npy')
    last_path = str(tmp_path / 'scan_T900K.npy')
    first = rows.filter(has_text='scan_T300K.npy')
    last = rows.filter(has_text='scan_T900K.npy')
    # Dropping in the upper half inserts before the target row.
    last.scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    start = last.bounding_box()
    end = first.bounding_box()
    page.mouse.move(start['x'] + 15, start['y'] + start['height'] / 2)
    page.mouse.down()
    page.mouse.move(end['x'] + 15, end['y'] + 8, steps=15)
    end = first.bounding_box()
    page.mouse.move(end['x'] + 15, end['y'] + 8)
    assert chart.evaluate('c => c.data.map(t => ({name:t.name, line:t.line}))') == original
    page.mouse.up()
    expected = [original[2], original[0], original[1]]
    page.wait_for_function("""names => {
        const chart = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return JSON.stringify(chart?.data?.map(t => t.name)) === JSON.stringify(names);
    }""", arg=[c['name'] for c in expected])
    assert chart.evaluate('c => c.data.map(t => ({name:t.name, line:t.line}))') == expected
    assert rows.first.get_attribute('data-path') == last_path
    with page.expect_download() as download:
        page.get_by_role('button', name='Save config', exact=True).click()
    config_path = tmp_path / 'order.json'
    download.value.save_as(config_path)
    settings = json.loads(config_path.read_text())['settings']
    assert settings[f'files:{tmp_path}'] == [last_path, first_path, str(tmp_path / 'scan_T600K.npy')]
    assert settings['preview_index'] == 1
    # Keyboard reordering and a later load must use the same saved ordering.
    rows.first.focus()
    rows.first.press('Alt+ArrowDown')
    page.wait_for_function("""name => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')?.data?.[0]?.name === name""", arg=original[0]['name'])
    page.locator('input[type=file]').set_input_files(config_path)
    page.get_by_role('button', name='Load config', exact=True).click()
    page.wait_for_function("""name => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')?.data?.[0]?.name === name""", arg=original[2]['name'])
    assert rows.first.get_attribute('data-path') == last_path
    assert page.get_by_role('button', name='Move up', exact=True).count() == 0
    page.screenshot(path=str(tmp_path / 'file-order.png'), full_page=True)
