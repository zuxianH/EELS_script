"""Pick draft background boundaries from the preview's energy axis."""
import math
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

_BACKGROUND_CLICK_JS = Path(__file__).with_name('background_click.js').read_text()


def apply_boundary_click(state, clicked):
    if not isinstance(clicked, dict):
        return
    group = state.get('bg_pick_target')
    keys = state.get('bg_pick_targets', {}).get(group)
    if not keys:
        return
    step = state.get('bg_pick_step', 0)
    if step >= len(keys):
        return
    target = keys[step]
    if (clicked.get('target') != target
            or clicked.get('context') != state.get('bg_pick_context')):
        return
    energy = clicked.get('energy')
    if not isinstance(energy, (int, float)) or not math.isfinite(energy):
        return
    low, high = state['bg_pick_bounds']
    if not low <= energy <= high:
        state['bg_pick_error'] = 'Choose an energy within the recorded spectrum.'
        return
    if step == 1 and energy <= state[keys[0]]:
        state['bg_pick_error'] = 'Click to the right of the minimum to set the maximum.'
        return
    state.pop('bg_pick_error', None)
    state[target] = float(energy)
    if group.startswith('area:'):
        peak_id = group.split(':', 1)[1]
        state[f'bg_area_selected_{peak_id}'] = True
    if step + 1 < len(keys):
        state['bg_pick_step'] = step + 1
    else:
        state['bg_pick_completed'] = (group, state[keys[0]], state[keys[-1]])
        state['bg_pick_target'] = ''
        state['bg_pick_step'] = 0


def reset_boundary_pick():
    st.session_state['bg_pick_step'] = 0
    st.session_state.pop('bg_pick_error', None)
    st.session_state.pop('bg_pick_completed', None)


def _on_boundary_click():
    # Component callbacks run before widgets are recreated, so number inputs
    # can receive new values without changing already-instantiated widgets.
    apply_boundary_click(st.session_state,
                         st.session_state.get('background_boundary_picker', {}).get('clicked'))


def render_boundary_picker(*, context, target):
    bridge = components.component('eels_background_click', js=_BACKGROUND_CLICK_JS)
    bridge(key='background_boundary_picker', height=0,
           data=dict(context=context, target=target), on_clicked_change=_on_boundary_click)
