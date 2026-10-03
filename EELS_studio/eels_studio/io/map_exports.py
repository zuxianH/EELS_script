"""Numerical scan-map and angle-resolved exports with unchanged schemas."""
import io
import json

import numpy as np


def all_maps_npz_bytes(scan_maps, bins, base_metadata):
    metadata = dict(base_metadata, axes=["energy_map", "probe_x", "probe_y"], maps=bins)
    energies = [entry["bin_energies_mev"] for entry in bins]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, scan_maps=np.stack(scan_maps),
                        selected_energies_mev=np.array([float(np.mean(e)) for e in energies]),
                        energy_lo_mev=np.array([e[0] for e in energies]),
                        energy_hi_mev=np.array([e[-1] for e in energies]),
                        bin_count=np.array([len(e) for e in energies]),
                        metadata_json=np.array(json.dumps(metadata)))
    return buffer.getvalue()


def export_map(energy, pixels, intensity, metadata):
    output = io.BytesIO()
    np.savez_compressed(output, energy_mev=energy, pixel_offset=pixels, intensity=intensity,
                        metadata_json=np.array(json.dumps(metadata)))
    return output.getvalue()
