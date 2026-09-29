from unittest.mock import patch

import file_order


def test_reorder_preserves_detector_probe_and_rejects_stale_selection():
    state = {'files:test': ['a', 'b'], '_comparison_curve_paths': ['a', 'b', 'b'],
             'preview_index': 2, 'visualization': 'Spectra & detector'}
    with patch.object(file_order.st, 'session_state', state):
        file_order.apply_file_order('files:test', ['b', 'a'])
        assert state['files:test'] == ['b', 'a']
        assert state['preview_index'] == 1
        for invalid in (['b', 'b'], ['a'], ['a', 'c'], None, [[], 'b']):
            file_order.apply_file_order('files:test', invalid)
            assert state['files:test'] == ['b', 'a']
        state['drag'] = {'order': {'before': ['a', 'b'], 'paths': ['a', 'b']}}
        file_order._on_order_change('files:test', 'drag')
        assert state['files:test'] == ['b', 'a']
