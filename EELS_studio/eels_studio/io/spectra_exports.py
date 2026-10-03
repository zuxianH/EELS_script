"""Full-range linear spectrum exports, preserving the existing file schemas."""
import csv
import io
import json

import numpy as np


def export_csv(curves):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["label", "source_file", "dummy", "probe_x", "probe_y", "energy_meV", "intensity"])
    for curve in curves:
        for energy, intensity in zip(curve["energy"], curve["intensity"]):
            writer.writerow([curve["label"], curve["path"], curve["dummy"], curve["probe_x"],
                             curve["probe_y"], energy, intensity])
    return output.getvalue().encode("utf-8")


def export_npz(curves, settings):
    arrays = {f"curve_{i:03d}": np.column_stack((c["energy"], c["intensity"])) for i, c in enumerate(curves)}
    metadata = {"settings": settings, "curves": [
        {k: v for k, v in c.items() if k not in ("energy", "intensity")} for c in curves
    ]}
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    return output.getvalue()


def export_notebook_npy(curves):
    buffer = io.BytesIO()
    np.save(buffer, np.stack([np.column_stack((c["energy"], c["intensity"])) for c in curves]))
    return buffer.getvalue()
