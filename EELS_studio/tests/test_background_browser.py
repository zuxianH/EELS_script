"""Optional real-browser background layout and viewport regression checks."""
import os

import numpy as np
import pytest

from test_axis_browser import spectrum_page


def open_background(page):
    page.get_by_role('tab', name='Background', exact=True).click()
    panel = page.locator('.st-key-background_method_panel')
    if panel.locator('details').first.get_attribute('open') is None:
        panel.locator('summary').first.click()


def test_background_desktop_narrow_and_preserved_view(spectrum_page):
    page = spectrum_page
    chart = page.locator(".st-key-spectrum_axis_target .js-plotly-plot")
    page.wait_for_function("""() => {
        const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return c?._ev?.listenerCount('plotly_relayout') > 0;
    }""")
    chart.evaluate("c => window.Plotly.relayout(c, {'xaxis.range': [20, 80], 'yaxis.range': [-4, -2]})")
    open_background(page)
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role("option", name="SNIP", exact=True).click()
    page.get_by_role("button", name="Preview", exact=True).click()
    page.get_by_text("solver: completed", exact=False).wait_for()
    preview = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    assert preview.evaluate("c => c._fullData.every(t => t.xaxis === 'x' && t.yaxis === 'y')")
    assert preview.evaluate("c => !c._fullLayout.xaxis2 && !c._fullLayout.yaxis2")
    assert preview.evaluate("c => c._fullData[1].line.dash") == "dash"
    page.get_by_role("button", name="Apply to all spectra", exact=True).click()
    page.get_by_text("Applied: SNIP", exact=False).last.wait_for()
    page.get_by_role("tab", name="Spectra", exact=True).click()
    page.wait_for_function("""() => {
        const c = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot');
        return c?._fullLayout?.yaxis.title.text.includes('Background-subtracted') ||
               c?._fullLayout?.yaxis.title.text.includes('background-subtracted');
    }""")
    page.wait_for_function("""() => {
        const l = document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')._fullLayout;
        return l.xaxis.range[0] === 20 && l.xaxis.range[1] === 80 && l.yaxis.range[0] === -4 && l.yaxis.range[1] === -2;
    }""")
    open_background(page)
    for width in (1700, 760, 390):
        page.set_viewport_size({"width": width, "height": 1100})
        page.wait_for_function("""() => {
            const block = document.querySelector('.st-key-background_layout');
            return block && block.getBoundingClientRect().width <= window.innerWidth;
        }""")
        # Inputs/actions must fit their containers without horizontal page scrolling.
        controls = page.locator('.st-key-background_layout input, .st-key-background_layout button')
        for control in controls.all():
            if control.is_visible():
                box = control.bounding_box()
                assert box["x"] >= -1 and box["x"] + box["width"] <= width + 1
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        if width <= 760:
            outer_row = page.locator('.st-key-background_layout [data-testid="stHorizontalBlock"]').first
            columns = outer_row.locator(':scope > [data-testid="stColumn"]')
            first, second = columns.nth(0).bounding_box(), columns.nth(1).bounding_box()
            assert second["y"] >= first["y"] + first["height"] - 1
        screenshot_dir = os.environ.get("EELS_BACKGROUND_SCREENSHOTS")
        if screenshot_dir:
            page.screenshot(path=f"{screenshot_dir}/background-{width}.png", full_page=True)
            preview.scroll_into_view_if_needed()
            page.screenshot(path=f"{screenshot_dir}/background-plot-{width}.png", full_page=True)


def test_background_shift_drag_shared_axes(spectrum_page):
    import math

    page = spectrum_page
    main = page.locator('.st-key-spectrum_axis_target .js-plotly-plot')
    main_before = main.evaluate('c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]')
    open_background(page)
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name='SNIP', exact=True).click()
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._ev?.listenerCount('plotly_relayout') > 0;
    }''')

    def ranges():
        return chart.evaluate('c => Object.fromEntries(["xaxis", "yaxis"].map(a => [a, [...c._fullLayout[a].range]]))')

    def gesture(axis, distance, shift=True):
        group = '.xy'
        horizontal = axis.startswith('x')
        handle = chart.locator(f'.draglayer {group} ' + ('.ewdrag' if horizontal else '.nsdrag'))
        handle.scroll_into_view_if_needed()
        box = handle.bounding_box()
        x, y = box['x'] + box['width'] / 2, box['y'] + box['height'] / 2
        if shift:
            page.keyboard.down('Shift')
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x + distance if horizontal else x,
                        y if horizontal else y - distance, steps=10)
        page.mouse.up()
        if shift:
            page.keyboard.up('Shift')

    def wait_ranges(expected):
        page.wait_for_function('''expected => {
            const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
            return Object.entries(expected).every(([axis, range]) =>
                c?._fullLayout?.[axis]?.range.every((v, i) => Math.abs(v - range[i]) < 1e-8));
        }''', arg=expected)

    initial = ranges()
    revision = chart.evaluate('c => c._fullLayout.meta.eels_render_revision')
    data_before = chart.evaluate('c => c._fullData.map(t => Array.from(t.y))')
    bounds_before = [page.get_by_role('spinbutton', name=name, exact=True).input_value()
                     for name in ('Fit energy minimum (meV)', 'Fit energy maximum (meV)')]
    for axis in ('yaxis', 'xaxis'):
        for distance in (45, -45):
            before = ranges()
            length = chart.evaluate('(c, axis) => c._fullLayout[axis]._length', axis)
            midpoint = sum(before[axis]) / 2
            half_span = (before[axis][1] - before[axis][0]) / 2
            half_span *= math.exp(-distance / length * math.log(4))
            expected = {**before, axis: [midpoint - half_span, midpoint + half_span]}
            gesture(axis, distance)
            wait_ranges(expected)

    # Native pan remains available and moves only the chosen intensity axis.
    before = ranges()
    gesture('yaxis', 35, shift=False)
    page.wait_for_function('''old => {
        const r = document.querySelector('.st-key-background_preview_plot .js-plotly-plot')._fullLayout.yaxis.range;
        return Math.abs(r[0] - old[0]) > 1e-8;
    }''', arg=before['yaxis'])
    after = ranges()
    assert after['yaxis'][1] - after['yaxis'][0] == pytest.approx(before['yaxis'][1] - before['yaxis'][0])
    assert after['xaxis'] == pytest.approx(before['xaxis'])
    assert chart.evaluate('c => c._fullLayout.meta.eels_render_revision') == revision
    np.testing.assert_array_equal(
        chart.evaluate('c => c._fullData.map(t => Array.from(t.y))'), data_before)
    assert [page.get_by_role('spinbutton', name=name, exact=True).input_value()
            for name in ('Fit energy minimum (meV)', 'Fit energy maximum (meV)')] == bounds_before

    # A rerun retains both axis ranges, with no interference from Spectra.
    page.get_by_role('tab', name='Spectra', exact=True).click()
    assert main.evaluate('c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]') == main_before
    page.get_by_text('Show hover details', exact=True).click()
    page.wait_for_function('''() => document.querySelector('.st-key-spectrum_axis_target .js-plotly-plot')?._fullLayout?.hovermode === false''')
    open_background(page)
    wait_ranges(after)
    chart.locator('[data-title="Reset axes"]').click()
    wait_ranges(initial)


def test_background_weighted_input_refits_and_preserves_energy_zoom(spectrum_page):
    page = spectrum_page
    open_background(page)
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name='SNIP', exact=True).click()
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._fullData?.length === 3 && c?._ev?.listenerCount('plotly_relayout') > 0;
    }''')
    linear_x, linear_y = chart.evaluate('c => [Array.from(c._fullData[0].x), Array.from(c._fullData[0].y)]')
    chart.evaluate("c => window.Plotly.relayout(c, {'xaxis.range': [10, 120]})")
    page.locator('.st-key-bg_display [role="combobox"]').click()
    page.get_by_role('option', name='Intensity × E²', exact=True).click()
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._fullLayout?.yaxis.title.text.includes('E²') && c?._fullData?.length === 1;
    }''')
    weighted_x, weighted_y = chart.evaluate('c => [Array.from(c._fullData[0].x), Array.from(c._fullData[0].y)]')
    np.testing.assert_array_equal(weighted_x, linear_x)
    np.testing.assert_allclose(weighted_y, np.asarray(linear_y) * np.square(linear_x))
    assert chart.evaluate('c => c._fullLayout.xaxis.range') == [10, 120]
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    page.wait_for_function('''() => document.querySelector('.st-key-background_preview_plot .js-plotly-plot')?._fullData?.length === 3''')
    data = chart.evaluate('c => c._fullData.map(t => Array.from(t.y))')
    np.testing.assert_allclose(data[0], weighted_y)
    finite = np.isfinite(data[2])
    np.testing.assert_allclose(np.asarray(data[0])[finite] - np.asarray(data[1])[finite], np.asarray(data[2])[finite])
    page.locator('.st-key-bg_display [role="combobox"]').click()
    page.get_by_role('option', name='Linear', exact=True).click()
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._fullLayout?.yaxis.title.text === 'Intensity' && c?._fullData?.length === 1;
    }''')
    np.testing.assert_array_equal(chart.evaluate('c => Array.from(c._fullData[0].y)'), linear_y)
    assert chart.evaluate('c => c._fullLayout.xaxis.range') == [10, 120]


@pytest.mark.parametrize("method", ["SNIP", "arPLS"])
def test_preview_interval_edit_replaces_old_boundary(spectrum_page, tmp_path, method):
    page = spectrum_page
    # Replace the fixture's temporary scan with enough bins for both intervals.
    energy = np.arange(600.)
    y = 2 + .001 * energy + .05 * np.sin(energy / 3) + np.exp(-((energy - 520) / 5) ** 2)
    np.save(tmp_path / 'scan_T300K.npy', y[:, None, None])
    page.get_by_role('button', name='Refresh files', exact=True).click()
    open_background(page)
    if method == "SNIP":
        page.locator('.st-key-bg_method [role="combobox"]').click()
        page.get_by_role('option', name='SNIP', exact=True).click()
    else:
        log_lambda = page.get_by_role('spinbutton', name='log10(λ), numeric', exact=True)
        log_lambda.fill('2.1')
        log_lambda.press('Enter')
    minimum = page.get_by_role('spinbutton', name='Fit energy minimum (meV)', exact=True)
    maximum = page.get_by_role('spinbutton', name='Fit energy maximum (meV)', exact=True)
    minimum.fill('190')
    minimum.press('Enter')
    maximum.fill('250')
    maximum.press('Enter')
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('Requested: 190–250 meV', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    chart.evaluate("c => window.Plotly.relayout(c, {'xaxis.range': [160, 270]})")
    # Match editing an input and clicking Preview directly, without pressing Enter.
    minimum.fill('150')
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('Requested: 150–250 meV', exact=False).wait_for()
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    try:
        page.wait_for_function('''() => {
            const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
            const rects = c?._fullLayout?.shapes.filter(s => s.type === 'rect');
            return rects?.length === 1 && rects.every(s => s.x0 >= 150 && s.x0 < 151 && s.x1 <= 250);
        }''', timeout=5000)
    except PlaywrightTimeoutError:
        details = chart.evaluate('''c => ({
            shapes: c._fullLayout.shapes.map(s => [s.type, s.x0, s.x1]),
            inputShapes: c.layout.shapes.map(s => [s.type, s.x0, s.x1]),
            starts: c._fullData.map(t => t.x[Array.from(t.y).findIndex(Number.isFinite)]),
            ranges: [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range],
            meta: c._fullLayout.meta
        })''')
        pytest.fail(f'Plot did not reflect the new interval: {details}')
    starts = chart.evaluate('''c => c._fullData.slice(1).map(t => {
        const j = Array.from(t.y).findIndex(Number.isFinite);
        return t.x[j];
    })''')
    assert len(starts) == 2 and all(150 <= value < 151 for value in starts)
    assert chart.evaluate('c => c._fullLayout.xaxis.range') == pytest.approx([160, 270])


def test_focus_interval_rescales_intensity_without_refitting(spectrum_page, tmp_path):
    page = spectrum_page
    y = 2 + 0.1 * np.sin(np.arange(600.) / 4)
    y[300] = 1e6  # dominant zero-loss bin outside the selected viewing interval
    np.save(tmp_path / 'scan_T300K.npy', y[:, None, None])
    page.get_by_role('button', name='Refresh files', exact=True).click()
    open_background(page)
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name='SNIP', exact=True).click()
    for label, value in [('Fit energy minimum (meV)', '150'), ('Fit energy maximum (meV)', '250')]:
        field = page.get_by_role('spinbutton', name=label, exact=True)
        field.fill(value)
        field.press('Enter')
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')

    def wait_focus():
        page.wait_for_function('''() => {
            const l = document.querySelector('.st-key-background_preview_plot .js-plotly-plot')?._fullLayout;
            return l?.xaxis.range[0] === 150 && l.xaxis.range[1] === 250 &&
                l.yaxis.range[1] < 1e-4 && l.yaxis.range[0] < 0 && l.yaxis.range[1] > 0;
        }''')

    wait_focus()
    data = chart.evaluate('c => c._fullData.map(t => Array.from(t.y))')
    page.wait_for_function('''() => document.querySelector('.st-key-background_preview_plot .js-plotly-plot')?._ev?.listenerCount('plotly_relayout') > 0''')
    chart.evaluate("c => window.Plotly.relayout(c, {'xaxis.range': [190, 230], 'yaxis.range': [0, 0.1]})")
    page.get_by_role('button', name='Focus fit interval', exact=True).click()
    wait_focus()
    np.testing.assert_array_equal(chart.evaluate('c => c._fullData.map(t => Array.from(t.y))'), data)
    page.get_by_role('button', name='Show full spectrum', exact=True).click()
    page.wait_for_function('''() => {
        const l = document.querySelector('.st-key-background_preview_plot .js-plotly-plot')?._fullLayout;
        return l?.xaxis.range[0] < -270 && l.xaxis.range[1] > 270 && l.yaxis.range[1] > 0.9;
    }''')
    np.testing.assert_array_equal(chart.evaluate('c => c._fullData.map(t => Array.from(t.y))'), data)
    page.get_by_role('button', name='Focus fit interval', exact=True).click()
    wait_focus()
    assert page.get_by_role('spinbutton', name='Fit energy minimum (meV)', exact=True).input_value() == '150.000000'
    assert page.get_by_role('spinbutton', name='Fit energy maximum (meV)', exact=True).input_value() == '250.000000'


@pytest.mark.parametrize('method, boundaries', [
    ('SNIP', [('Fit interval', 'Fit energy minimum (meV)', .2),
              ('Fit interval', 'Fit energy maximum (meV)', .8)]),
    ('power0', [('Segment 1', 'Segment 1 min (meV)', .1),
                ('Segment 1', 'Segment 1 max (meV)', .3),
                ('Segment 2', 'Segment 2 min (meV)', .6),
                ('Segment 2', 'Segment 2 max (meV)', .85)]),
])
def test_pick_background_bounds_with_plot_clicks(spectrum_page, method, boundaries):
    from playwright.sync_api import expect

    page = spectrum_page
    open_background(page)
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name=method, exact=True).click()
    # Picking uses energy coordinates in either fitting intensity form.
    page.locator('.st-key-bg_display [role="combobox"]').click()
    page.get_by_role('option', name='Intensity × E²', exact=True).click()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._fullLayout?.yaxis.title.text.includes('E²') &&
            c?._ev?.listenerCount('plotly_relayout') > 0;
    }''')
    ranges = chart.evaluate('c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]')
    for index, (label, input_label, fraction) in enumerate(boundaries):
        if index % 2 == 0:
            page.get_by_role('button', name=f'Pick {label}', exact=True).click()
        instruction = 'First click: set the minimum.' if index % 2 == 0 else 'Second click: set the maximum.'
        page.get_by_text(f'{label} — {instruction}', exact=False).wait_for()
        handle = chart.locator('.draglayer .xy .nsewdrag')
        handle.scroll_into_view_if_needed()
        box = handle.bounding_box()
        expected_energy = ranges[0][0] + fraction * (ranges[0][1] - ranges[0][0])
        page.mouse.click(box['x'] + box['width'] * fraction, box['y'] + box['height'] * .65)
        field = page.get_by_role('spinbutton', name=input_label, exact=True)
        page.wait_for_function('''({label, energy}) => {
            const input = document.querySelector(`input[aria-label="${label}"]`);
            return input && Math.abs(Number(input.value) - energy) < 0.6;
        }''', arg=dict(label=input_label, energy=expected_energy))
        assert float(field.input_value()) == pytest.approx(expected_energy, abs=.6)
        if index % 2 == 0:
            page.get_by_text(f'{label} — Second click: set the maximum.', exact=False).wait_for()
        else:
            page.get_by_text(f'{label} set:', exact=False).wait_for()
        actual_ranges = chart.evaluate('c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]')
        np.testing.assert_allclose(actual_ranges, ranges)
    values = [page.get_by_role('spinbutton', name=name, exact=True).input_value()
              for _, name, _ in boundaries]
    # Start another pair, then verify panning does not count as the first click.
    page.get_by_role('button', name=f'Pick {boundaries[-1][0]}', exact=True).click()
    page.get_by_text('First click: set the minimum.', exact=False).wait_for()
    handle = chart.locator('.draglayer .xy .nsewdrag')
    handle.scroll_into_view_if_needed()
    box = handle.bounding_box()
    page.mouse.move(box['x'] + box['width'] * .3, box['y'] + box['height'] * .3)
    page.mouse.down()
    page.mouse.move(box['x'] + box['width'] * .7, box['y'] + box['height'] * .7, steps=8)
    page.mouse.up()
    page.wait_for_function('''before => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return Math.abs(c._fullLayout.xaxis.range[0] - before) > 1;
    }''', arg=ranges[0][0])
    panned = chart.evaluate('c => [c._fullLayout.xaxis.range, c._fullLayout.yaxis.range]')
    np.testing.assert_allclose(np.diff(panned, axis=1), np.diff(ranges, axis=1))
    for (_, name, _), value in zip(boundaries, values):
        expect(page.get_by_role('spinbutton', name=name, exact=True)).to_have_value(value)
    page.get_by_role('button', name=f'Cancel {boundaries[-1][0]}', exact=True).click()
    expect(page.get_by_text('First click: set the minimum.', exact=False)).to_have_count(0)
    expect(page.get_by_text('Second click: set the maximum.', exact=False)).to_have_count(0)
    handle.click(position=dict(x=30, y=30))
    for (_, name, _), value in zip(boundaries, values):
        expect(page.get_by_role('spinbutton', name=name, exact=True)).to_have_value(value)
    if method == 'SNIP':
        domain = page.locator('.st-key-bg_domain [role="combobox"]')
        domain.focus()
        domain.press('ArrowDown')
        page.get_by_role('option', name='Full recorded spectrum', exact=True).click()
        expect(page.get_by_role('button', name='Pick Fit interval', exact=True)).to_be_disabled()
        expect(page.locator('.st-key-background_peak_1').get_by_role('button', name='Pick center', exact=True)).to_be_visible()


def test_corrected_area_recalculates_for_weighted_fit(spectrum_page):
    from playwright.sync_api import expect

    page = spectrum_page
    open_background(page)
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name='SNIP', exact=True).click()
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._fullData?.length === 3 && c?._ev?.listenerCount('plotly_relayout') > 0;
    }''')
    traces = chart.evaluate('c => c._fullData.map(t => ({x: Array.from(t.x), y: Array.from(t.y)}))')
    fit_bounds = [page.get_by_role('spinbutton', name=name, exact=True).input_value()
                  for name in ('Fit energy minimum (meV)', 'Fit energy maximum (meV)')]
    panel = page.locator('.st-key-background_peak_1')
    width = panel.get_by_role('spinbutton', name='Half-width (± meV)', exact=True)
    width.fill('10')
    width.press('Enter')
    panel.get_by_role('button', name='Pick center', exact=True).click()
    page.get_by_text('Peak 1 center — click once in the plot.', exact=False).wait_for()
    handle = chart.locator('.draglayer .xy .nsewdrag')
    handle.scroll_into_view_if_needed()
    box = handle.bounding_box()
    x_range = chart.evaluate('c => c._fullLayout.xaxis.range')
    expected_center = x_range[0] + .6 * (x_range[1] - x_range[0])
    page.mouse.click(box['x'] + box['width'] * .6, box['y'] + box['height'] * .6)
    page.wait_for_function('''center => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        const s = c?.layout?.shapes?.find(s => s.name === 'Peak 1 center');
        return s && Math.abs(s.x0 - center) < .6;
    }''', arg=expected_center)
    expect(page.get_by_text('Peak 1 center — click once in the plot.', exact=False)).to_have_count(0)
    area_text = panel.get_by_text('Area: ', exact=False).first
    normalized_text = panel.get_by_text('Normalized area: ', exact=False)
    normalized_text.wait_for()
    bounds = chart.evaluate("c => {const s = c.layout.shapes.find(s => s.name === 'Peak 1 area'); return [s.x0, s.x1];}")
    assert bounds[1] - bounds[0] == pytest.approx(20.)
    center = chart.evaluate("c => c.layout.shapes.find(s => s.name === 'Peak 1 center').x0")
    assert bounds == pytest.approx([center - 10, center + 10])
    # Independently integrate the rendered corrected trace over the chosen bounds.
    x, y = np.array(traces[2]['x']), np.array(traces[2]['y'], dtype=float)
    inside = (x > bounds[0]) & (x < bounds[1])
    selected_x = np.r_[bounds[0], x[inside], bounds[1]]
    selected_y = np.interp(selected_x, x, y)
    expected = sum((b - a) * (u + v) / 2 for a, b, u, v in
                   zip(selected_x[:-1], selected_x[1:], selected_y[:-1], selected_y[1:]))
    assert float(area_text.inner_text().split(': ', 1)[1].split(' intensity')[0]) == pytest.approx(expected, rel=1e-4, abs=1e-12)
    original_x, original_y = np.array(traces[0]['x']), np.array(traces[0]['y'])
    original_total = sum(np.diff(original_x) * (original_y[:-1] + original_y[1:]) / 2)
    assert float(normalized_text.inner_text().split(': ', 1)[1]) == pytest.approx(expected / original_total, rel=1e-4, abs=1e-12)
    assert [page.get_by_role('spinbutton', name=name, exact=True).input_value()
            for name in ('Fit energy minimum (meV)', 'Fit energy maximum (meV)')] == fit_bounds
    after = chart.evaluate('c => c._fullData.map(t => ({x: Array.from(t.x), y: Array.from(t.y)}))')
    for actual, original in zip(after, traces):
        np.testing.assert_array_equal(actual['x'], original['x'])
        np.testing.assert_array_equal(actual['y'], original['y'])
    assert chart.evaluate('c => c.layout.shapes.some(s => s.name === "Peak 1 area")')
    # Width changes expand both sides equally without moving the chosen center.
    width.fill('20')
    width.press('Enter')
    page.wait_for_function('''center => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        const s = c?.layout?.shapes?.find(s => s.name === 'Peak 1 area');
        return s && Math.abs(s.x0 - (center - 20)) < 1e-8 && Math.abs(s.x1 - (center + 20)) < 1e-8;
    }''', arg=center)
    before, normalized_before = area_text.inner_text(), normalized_text.inner_text()
    method_panel = page.locator('.st-key-background_method_panel')
    area_panel = page.locator('.st-key-background_area_panel')
    expect(panel.get_by_role('spinbutton', name='Center (meV)', exact=True)).to_be_visible()
    expect(area_panel.locator('[data-testid="stMetric"]')).to_have_count(0)
    expect(area_panel.get_by_text('Normalized area =', exact=False)).to_have_count(0)
    method_panel.locator('summary').first.click()
    expect(page.locator('.st-key-bg_method [role="combobox"]')).to_be_hidden()
    area_panel.locator('summary').first.click()
    expect(width).to_be_hidden()
    expect(area_text).to_be_hidden()
    expect(chart).to_be_visible()
    area_panel.locator('summary').first.click()
    expect(width).to_have_value('20')
    expect(area_text).to_have_text(before)
    screenshot_dir = os.environ.get('EELS_BACKGROUND_SCREENSHOTS')
    if screenshot_dir:
        page.locator('.st-key-background_layout').screenshot(path=f'{screenshot_dir}/background-controls.png', animations='disabled')
    page.locator('.st-key-bg_display [role="combobox"]').click()
    page.get_by_role('option', name='Intensity × E²', exact=True).click()
    page.wait_for_function('''() => document.querySelector('.st-key-background_preview_plot .js-plotly-plot')?._fullLayout?.yaxis.title.text.includes('E²') ''')
    expect(panel.get_by_text('Press Preview to fit the background before calculating corrected area.')).to_be_visible()
    method_panel.locator('summary').first.click()
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    expect(area_text).to_be_visible()
    expect(normalized_text).to_be_visible()
    assert area_text.inner_text() != before
    assert normalized_text.inner_text() != normalized_before
    weighted_traces = chart.evaluate('c => c._fullData.map(t => ({x: Array.from(t.x), y: Array.from(t.y)}))')
    weighted_x = np.asarray(weighted_traces[2]['x'])
    weighted_corrected = np.asarray(weighted_traces[2]['y'], dtype=float)
    weighted_bounds = chart.evaluate("c => {const s = c.layout.shapes.find(s => s.name === 'Peak 1 area'); return [s.x0, s.x1];}")
    inside = (weighted_x > weighted_bounds[0]) & (weighted_x < weighted_bounds[1])
    selected_x = np.r_[weighted_bounds[0], weighted_x[inside], weighted_bounds[1]]
    selected_y = np.interp(selected_x, weighted_x, weighted_corrected)
    weighted_area = sum(np.diff(selected_x) * (selected_y[:-1] + selected_y[1:]) / 2)
    weighted_input = np.asarray(weighted_traces[0]['y'], dtype=float)
    weighted_total = sum(np.diff(weighted_x) * (weighted_input[:-1] + weighted_input[1:]) / 2)
    assert float(area_text.inner_text().split(': ', 1)[1].split(' intensity')[0]) == pytest.approx(weighted_area, rel=1e-4)
    assert float(normalized_text.inner_text().split(': ', 1)[1]) == pytest.approx(weighted_area / weighted_total, rel=1e-4)
    # A stale corrected preview must not leave a stale area visible.
    field = page.get_by_role('spinbutton', name='Maximum half-window (meV)', exact=True)
    field.fill('20')
    field.press('Enter')
    expect(panel.get_by_text('Press Preview to fit the background before calculating corrected area.')).to_be_visible()
    expect(panel.get_by_text('Area: ', exact=False)).to_have_count(0)
    expect(panel.get_by_text('Normalized area: ', exact=False)).to_have_count(0)


def test_multiple_peak_areas_update_independently(spectrum_page):
    from playwright.sync_api import expect

    page = spectrum_page
    open_background(page)
    page.locator('.st-key-bg_method [role="combobox"]').click()
    page.get_by_role('option', name='SNIP', exact=True).click()
    page.get_by_role('button', name='Preview', exact=True).click()
    page.get_by_text('solver: completed', exact=False).wait_for()
    chart = page.locator('.st-key-background_preview_plot .js-plotly-plot')
    page.wait_for_function('''() => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        return c?._fullData?.length === 3 && c?._ev?.listenerCount('plotly_relayout') > 0;
    }''')
    fit_bounds = [page.get_by_role('spinbutton', name=name, exact=True).input_value()
                  for name in ('Fit energy minimum (meV)', 'Fit energy maximum (meV)')]
    traces = chart.evaluate('c => c._fullData.map(t => Array.from(t.y))')
    peak1 = page.locator('.st-key-background_peak_1')
    peak2 = page.locator('.st-key-background_peak_2')
    expect(peak1).to_be_visible()
    expect(peak2).to_be_visible()
    initial_range = chart.evaluate('c => c._fullLayout.xaxis.range')
    for number, peak, half_width, fraction in ((1, peak1, 12, .35), (2, peak2, 18, .72)):
        field = peak.get_by_role('spinbutton', name='Half-width (± meV)', exact=True)
        field.fill(str(half_width))
        field.press('Enter')
        peak.get_by_role('button', name='Pick center', exact=True).click()
        peak.get_by_text(f'Peak {number} center — click once in the plot.', exact=False).wait_for()
        handle = chart.locator('.draglayer .xy .nsewdrag')
        handle.scroll_into_view_if_needed()
        box = handle.bounding_box()
        energy = initial_range[0] + fraction * (initial_range[1] - initial_range[0])
        page.mouse.click(box['x'] + box['width'] * fraction, box['y'] + box['height'] * .6)
        page.wait_for_function('''({number, energy}) => {
            const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
            const s = c?.layout?.shapes?.find(s => s.name === `Peak ${number} center`);
            return s && Math.abs(s.x0 - energy) < .6;
        }''', arg=dict(number=number, energy=energy))
        expect(peak.get_by_text('Normalized area: ', exact=False)).to_be_visible()
    peak1_area = peak1.get_by_text('Area: ', exact=False).first.inner_text()
    peak2_area = peak2.get_by_text('Area: ', exact=False).first.inner_text()
    first_range = chart.evaluate("c => {const s = c.layout.shapes.find(s => s.name === 'Peak 1 area'); return [s.x0, s.x1];}")
    second_range = chart.evaluate("c => {const s = c.layout.shapes.find(s => s.name === 'Peak 2 area'); return [s.x0, s.x1];}")
    assert first_range != second_range
    peak2.get_by_role('spinbutton', name='Half-width (± meV)', exact=True).fill('24')
    peak2.get_by_role('spinbutton', name='Half-width (± meV)', exact=True).press('Enter')
    page.wait_for_function('''old => {
        const c = document.querySelector('.st-key-background_preview_plot .js-plotly-plot');
        const s = c?.layout?.shapes?.find(s => s.name === 'Peak 2 area');
        return s && Math.abs((s.x1 - s.x0) - 48) < 1e-8;
    }''', arg=second_range)
    assert chart.evaluate("c => {const s = c.layout.shapes.find(s => s.name === 'Peak 1 area'); return [s.x0, s.x1];}") == pytest.approx(first_range)
    expect(peak1.get_by_text('Area: ', exact=False).first).to_have_text(peak1_area)
    page.get_by_role('button', name='Add peak', exact=True).click()
    peak3 = page.locator('.st-key-background_peak_3')
    expect(peak3).to_be_visible()
    peak3.get_by_role('spinbutton', name='Center (meV)', exact=True).fill('100')
    peak3.get_by_role('spinbutton', name='Center (meV)', exact=True).press('Enter')
    expect(peak3.get_by_text('Normalized area: ', exact=False)).to_be_visible()
    peak2.get_by_role('button', name='Remove', exact=True).click()
    expect(peak2).to_have_count(0)
    expect(peak3).to_be_visible()
    expect(peak3.get_by_text('Normalized area: ', exact=False)).to_be_visible()
    expect(peak1.get_by_text('Area: ', exact=False).first).to_have_text(peak1_area)
    assert [page.get_by_role('spinbutton', name=name, exact=True).input_value()
            for name in ('Fit energy minimum (meV)', 'Fit energy maximum (meV)')] == fit_bounds
    np.testing.assert_array_equal(chart.evaluate('c => c._fullData.map(t => Array.from(t.y))'), traces)


def test_peak_area_controls_use_compact_rows(spectrum_page):
    page = spectrum_page
    open_background(page)
    peak = page.locator('.st-key-background_peak_1')
    peak.get_by_role('spinbutton', name='Center (meV)', exact=True).wait_for()
    peak.get_by_role('spinbutton', name='Half-width (± meV)', exact=True).wait_for()
    peak.get_by_role('button', name='Pick center', exact=True).wait_for()
    for viewport_width in (1700, 390):
        page.set_viewport_size({'width': viewport_width, 'height': 1100})
        boxes = peak.evaluate("""card => {
            const center = card.querySelector('.st-key-bg_area_center_1 input');
            const width = card.querySelector('.st-key-bg_area_half_width_1 input');
            const buttons = [...card.querySelectorAll('button')];
            const pick = buttons.find(b => b.textContent.trim() === 'Pick center' && b.getBoundingClientRect().width > 0);
            const remove = buttons.find(b => b.textContent.trim() === 'Remove');
            return [center, width, pick, remove].map(e => e.getBoundingClientRect().toJSON());
        }""")
        center_box, half_width_box, pick_box, remove_box = boxes
        assert abs(center_box['y'] - half_width_box['y']) < 2, (viewport_width, boxes)
        assert center_box['right'] <= half_width_box['left']
        assert abs(pick_box['y'] - remove_box['y']) < 2, (viewport_width, boxes)
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
