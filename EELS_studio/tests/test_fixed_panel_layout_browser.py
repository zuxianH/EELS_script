"""Fixed Spectra layout must stay fixed when older custom layouts are loaded."""
import json

import pytest

from panel_layout import ADJUSTABLE_PANELS_ENABLED
from test_axis_browser import spectrum_page


@pytest.mark.skipif(ADJUSTABLE_PANELS_ENABLED, reason='Fixed layout is disabled')
def test_saved_custom_layout_does_not_enable_adjustable_panels(spectrum_page, tmp_path):
    page = spectrum_page
    with page.expect_download() as downloaded:
        page.get_by_role('button', name='Save config', exact=True).click()
    path = tmp_path / 'custom-layout.json'
    downloaded.value.save_as(path)
    config = json.loads(path.read_text())
    config['settings'].update(panel_detector_width=.65, panel_spectrum_height=800,
                              panel_detector_height=750, panel_detector_first=False)
    path.write_text(json.dumps(config))
    page.locator('input[type=file]').set_input_files(path)
    page.get_by_role('button', name='Load config', exact=True).click()
    page.get_by_role('spinbutton', name='radius', exact=True).wait_for()
    # Wait for the load to finish, not just for the previous charts to be visible.
    page.wait_for_function("""() => {
        const c = document.querySelector('.st-key-detector_click_target .js-plotly-plot');
        return String(c?._fullLayout?.xaxis?.uirevision).endsWith(':1');
    }""")
    assert page.locator('.eels-panel-divider, .eels-height-handle, .eels-movable-title').count() == 0
    assert page.get_by_role('button', name='Reset panel layout', exact=True).count() == 0
    assert page.locator('.st-key-panel_layout_bridge').count() == 0
    detector = page.locator('.st-key-detector_panel').bounding_box()
    spectrum = page.locator('.st-key-spectrum_panel').bounding_box()
    assert detector['x'] < spectrum['x']
    assert spectrum['width'] / detector['width'] == pytest.approx(2, abs=.1)
    for target, height in [('detector_click_target', 420), ('spectrum_axis_target', 520)]:
        chart = page.locator(f'.st-key-{target} .js-plotly-plot')
        assert chart.evaluate('c => c._fullLayout.height') == height
