"""Detector coordinate ranges must not grow while its panel is resized."""
import pytest

from test_axis_browser import spectrum_page
from test_panel_layout_browser import drag
from panel_layout import ADJUSTABLE_PANELS_ENABLED


def wait_ready(page):
    page.locator('.st-key-detector_click_target .js-plotly-plot').wait_for(state='visible')
    page.wait_for_function("""() => {
        const c = document.querySelector('.st-key-detector_click_target .js-plotly-plot');
        return c?._fullLayout?.xaxis && c?._fullLayout?.yaxis;
    }""")


def resize_workspace(page):
    if ADJUSTABLE_PANELS_ENABLED:
        drag(page, page.get_by_role('separator', name='Resize Detector height', exact=True), dy=100)
        drag(page, page.get_by_role('separator', name='Resize Detector and EELS Spectrum panels'), dx=80)
    else:
        for width in (1450, 1700):
            page.set_viewport_size({'width': width, 'height': 1300})
            page.wait_for_timeout(500)


def detector_state(page):
    return page.locator('.st-key-detector_click_target .js-plotly-plot').evaluate('''c => ({
        width: c._fullLayout.width, height: c._fullLayout.height,
        anchorWidth: c.closest('.st-key-detector_click_target').getBoundingClientRect().width,
        hostWidth: c.closest('[data-testid="stPlotlyChart"]').getBoundingClientRect().width,
        x: c._fullLayout.xaxis.range, y: c._fullLayout.yaxis.range,
        xdomain: c._fullLayout.xaxis.domain, ydomain: c._fullLayout.yaxis.domain,
        constrain: [c._fullLayout.xaxis.constrain, c._fullLayout.yaxis.constrain],
        pixelsPerUnit: [Math.abs(c._fullLayout.xaxis._m), Math.abs(c._fullLayout.yaxis._m)]
    })''')


def test_detector_does_not_zoom_out_after_panel_resize(spectrum_page):
    page = spectrum_page
    wait_ready(page)
    before = detector_state(page)
    assert before['x'] == pytest.approx([-.5, 13.5])
    assert before['y'] == pytest.approx([-.5, 11.5])
    for _ in range(2):
        resize_workspace(page)
    after = detector_state(page)
    assert after['x'] == pytest.approx(before['x'], rel=1e-6, abs=1e-6)
    assert after['y'] == pytest.approx(before['y'], rel=1e-6, abs=1e-6)
    # Resizing should settle; the image must not keep receding while idle.
    page.wait_for_timeout(1500)
    settled = detector_state(page)
    assert settled['x'] == pytest.approx(after['x'], rel=1e-6, abs=1e-6)
    assert settled['y'] == pytest.approx(after['y'], rel=1e-6, abs=1e-6)
    assert settled['pixelsPerUnit'][0] == pytest.approx(settled['pixelsPerUnit'][1], rel=1e-6)


def test_detector_manual_zoom_and_click_survive_resize(spectrum_page):
    page = spectrum_page
    wait_ready(page)
    chart = page.locator('.st-key-detector_click_target .js-plotly-plot')
    chart.locator('.nsewdrag').hover()
    page.mouse.wheel(0, -160)
    page.wait_for_function("""() => {
        const x = document.querySelector('.st-key-detector_click_target .js-plotly-plot')._fullLayout.xaxis.range;
        return x[1] - x[0] < 13;
    }""")
    # Wheel zoom has a short animation/debounce before emitting its final range.
    page.wait_for_timeout(500)
    zoomed = detector_state(page)
    resize_workspace(page)
    after = detector_state(page)
    assert after['x'] == pytest.approx(zoomed['x'], rel=1e-6, abs=1e-6)
    assert after['y'] == pytest.approx(zoomed['y'], rel=1e-6, abs=1e-6)
    chart.scroll_into_view_if_needed()
    point = chart.evaluate("""c => {
        const box = c.getBoundingClientRect(), l = c._fullLayout;
        return {x: box.x + l.xaxis._offset + l.xaxis.l2p(6),
                y: box.y + l.yaxis._offset + l.yaxis.l2p(5)};
    }""")
    page.mouse.click(point['x'], point['y'])
    page.wait_for_function("""() => {
        const inputs = [...document.querySelectorAll('input[type=number]')];
        return ['px','py'].every(name => inputs.find(i => i.getAttribute('aria-label') === name)?.value === '-1');
    }""")
    after_click = detector_state(page)
    assert after_click['x'] == pytest.approx(zoomed['x'], rel=1e-6, abs=1e-6)
    assert after_click['y'] == pytest.approx(zoomed['y'], rel=1e-6, abs=1e-6)


def test_legacy_expanded_detector_view_is_discarded(spectrum_page):
    page = spectrum_page
    wait_ready(page)
    page.evaluate("""() => {
        window[Symbol.for('eels.detector.viewport')] = {
            xaxis: {range: [-1e9, 1e9], revision: 'detector:(12, 14)'},
            yaxis: {range: [-1e9, 1e9], revision: 'detector:(12, 14)'}
        };
    }""")
    page.get_by_label('Show detector preview', exact=True).locator('xpath=ancestor::label').click()
    page.locator('.st-key-detector_panel').wait_for(state='detached')
    page.get_by_label('Show detector preview', exact=True).locator('xpath=ancestor::label').click()
    wait_ready(page)
    restored = detector_state(page)
    assert restored['x'] == pytest.approx([-.5, 13.5])
    assert restored['y'] == pytest.approx([-.5, 11.5])
