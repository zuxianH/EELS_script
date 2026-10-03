"""Stable source identities, shared by views and exports."""
import json


def curve_identity_key(curve):
    """Stable identifier for a curve's source, independent of its display label."""
    return json.dumps([curve["path"], curve["dummy"], curve["probe_x"], curve["probe_y"]])
