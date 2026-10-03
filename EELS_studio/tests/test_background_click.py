"""Two clicks set draft interval endpoints without applying a fit."""
import pytest

from eels_studio.ui.components.background_click import apply_boundary_click


def state_for(target='fit'):
    return dict(bg_pick_target=target, bg_pick_context='spectrum:SNIP', bg_pick_step=0,
                bg_pick_targets={'fit': ('bg_min', 'bg_max'),
                                 'segment_1': ('bg_seg_lo_1', 'bg_seg_hi_1')},
                bg_pick_bounds=(0., 200.), bg_min=10., bg_max=180.,
                bg_seg_lo_1=90., bg_seg_hi_1=150.)


def click(state, key, energy):
    apply_boundary_click(state, dict(context='spectrum:SNIP', target=key, energy=energy))


@pytest.mark.parametrize('group, low_key, high_key', [
    ('fit', 'bg_min', 'bg_max'), ('segment_1', 'bg_seg_lo_1', 'bg_seg_hi_1'),
])
def test_two_clicks_set_minimum_then_maximum_and_finish(group, low_key, high_key):
    state = state_for(group)
    old_max = state[high_key]
    click(state, low_key, 35.25)
    assert state[low_key] == 35.25
    assert state[high_key] == old_max
    assert state['bg_pick_step'] == 1
    assert state['bg_pick_target'] == group
    click(state, high_key, 170.)
    assert state[low_key] == 35.25
    assert state[high_key] == 170.
    assert state['bg_pick_target'] == ''
    click(state, low_key, 50.)
    assert state[low_key] == 35.25


@pytest.mark.parametrize('changes', [
    {'context': 'previous spectrum'}, {'target': 'bg_max'}, {'energy': float('nan')},
    {'energy': float('inf')}, {'energy': -1.}, {'energy': 201.}, {'energy': None},
])
def test_stale_or_invalid_click_does_not_change_bounds(changes):
    state = state_for()
    event = dict(context='spectrum:SNIP', target='bg_min', energy=35.)
    event.update(changes)
    apply_boundary_click(state, event)
    assert (state['bg_min'], state['bg_max']) == (10., 180.)
    assert state['bg_pick_step'] == 0


@pytest.mark.parametrize('maximum', [20., 35.])
def test_maximum_must_be_above_first_click(maximum):
    state = state_for()
    click(state, 'bg_min', 35.)
    click(state, 'bg_max', maximum)
    assert state['bg_max'] == 180.
    assert state['bg_pick_step'] == 1
    assert state['bg_pick_error']
    click(state, 'bg_max', 160.)
    assert state['bg_max'] == 160.
    assert 'bg_pick_error' not in state


def test_delayed_first_click_cannot_overwrite_second_step():
    state = state_for()
    click(state, 'bg_min', 35.)
    click(state, 'bg_min', 50.)
    assert state['bg_min'] == 35.
    assert state['bg_max'] == 180.
    assert state['bg_pick_step'] == 1


def test_disabled_picking_ignores_click():
    state = state_for('')
    click(state, 'bg_min', 35.)
    assert state['bg_min'] == 10.


def test_area_center_click_keeps_width_and_fit_bounds():
    state = state_for('area:1')
    state['bg_pick_targets']['area:1'] = ('bg_area_center_1',)
    state.update(bg_area_center_1=65., bg_area_half_width_1=15.)
    click(state, 'bg_area_center_1', 80.)
    assert (state['bg_area_center_1'], state['bg_area_half_width_1']) == (80., 15.)
    assert (state['bg_min'], state['bg_max']) == (10., 180.)
    assert (state['bg_seg_lo_1'], state['bg_seg_hi_1']) == (90., 150.)
    assert state['bg_area_selected_1'] is True
    assert state['bg_pick_target'] == ''
    click(state, 'bg_area_center_1', 90.)
    assert state['bg_area_center_1'] == 80.


def test_picking_two_peak_centers_keeps_both_ranges_independent():
    state = state_for('area:1')
    state['bg_pick_targets'].update({'area:1': ('bg_area_center_1',),
                                     'area:2': ('bg_area_center_2',)})
    state.update(bg_area_center_1=50., bg_area_center_2=90.,
                 bg_area_half_width_1=8., bg_area_half_width_2=12.)
    click(state, 'bg_area_center_1', 55.)
    assert state['bg_area_selected_1'] is True
    assert state['bg_area_center_1'] == 55.
    assert state['bg_area_center_2'] == 90.
    assert state['bg_area_half_width_2'] == 12.
    state['bg_pick_target'] = 'area:2'
    click(state, 'bg_area_center_2', 110.)
    assert state['bg_area_selected_2'] is True
    assert state['bg_area_center_2'] == 110.
    assert state['bg_area_center_1'] == 55.
    assert state['bg_area_half_width_1'] == 8.
    assert (state['bg_min'], state['bg_max']) == (10., 180.)
