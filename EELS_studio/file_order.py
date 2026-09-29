"""Drag-to-reorder comparison files, committing only on drop."""
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components


def apply_file_order(files_key, ordered):
    selected = st.session_state.get(files_key, [])
    if (not isinstance(ordered, list) or any(not isinstance(p, str) for p in ordered)
            or len(ordered) != len(selected) or set(ordered) != set(selected)):
        return
    # Keep the same detector spectrum, including its probe position for 6D data.
    curve_paths = st.session_state.get('_comparison_curve_paths', [])
    preview = st.session_state.get('preview_index', 0)
    if (st.session_state.get('visualization') == 'Spectra & detector'
            and set(curve_paths) == set(selected) and 0 <= preview < len(curve_paths)):
        positions = {p: i for i, p in enumerate(ordered)}
        reordered = sorted(range(len(curve_paths)), key=lambda i: positions[curve_paths[i]])
        st.session_state['preview_index'] = reordered.index(preview)
    st.session_state[files_key] = ordered


def _on_order_change(files_key, component_key):
    event = st.session_state.get(component_key, {}).get('order')
    # Ignore a delayed drop if the selection changed in the meantime.
    if isinstance(event, dict) and event.get('before') == st.session_state.get(files_key):
        apply_file_order(files_key, event.get('paths'))


def render_file_order(files_key, selected):
    if len(selected) < 2:
        return
    with st.expander('File order', expanded=True):
        st.caption('Drag files into order for the plot, legend, and exports. Keyboard: Alt + ↑ / ↓.')
        component_key = f'file_order_drag:{files_key}'
        component = components.component('eels_file_order', js=Path(__file__).with_suffix('.js').read_text())
        component(key=component_key, height=min(288, 48 * len(selected)),
                  data={'items': [{'path': p, 'label': Path(p).name} for p in selected]},
                  on_order_change=lambda: _on_order_change(files_key, component_key))
