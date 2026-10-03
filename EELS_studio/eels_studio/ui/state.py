"""Background workspace lifecycle and input invalidation, independent of widget rendering."""
from dataclasses import dataclass, field
import hashlib
import json
from typing import Callable

import numpy as np

from eels_studio.core.identity import curve_identity_key
from eels_studio.core.background import BackgroundConfig, BackgroundResult, fit_background


def input_fingerprints(curves, settings, revisions):
    """Labels, styles and view limits are deliberately excluded."""
    processing = {k: v for k, v in settings.items() if k not in ("plot", "probe_positions_xy")}
    result = {}
    for curve in curves:
        identity = curve_identity_key(curve)
        digest = hashlib.sha256(json.dumps([identity, processing, revisions[curve["path"]]], sort_keys=True).encode())
        for name in ("energy", "intensity"):
            digest.update(np.asarray(curve[name], dtype="<f8").tobytes())
        result[identity] = digest.hexdigest()
    return result


@dataclass
class BackgroundState:
    draft: BackgroundConfig | None = None
    preview: BackgroundResult | None = None
    preview_key: tuple | None = None
    applied_config: BackgroundConfig | None = None
    applied_results: dict[str, BackgroundResult] = field(default_factory=dict)
    fingerprints: dict[str, str] = field(default_factory=dict)
    source_revisions: dict[str, tuple[int, int]] = field(default_factory=dict)
    signal: str = "Input"

    def sync_inputs(self, fingerprints):
        changed = self.fingerprints != fingerprints
        if changed:
            self.reset()
            self.fingerprints = fingerprints.copy()
        return changed

    def reset(self):
        self.preview = None
        self.preview_key = None
        self.applied_config = None
        self.applied_results = {}
        self.signal = "Input"

    def apply(self, curves, config, fitter: Callable = fit_background):
        results, failures = {}, {}
        for curve in curves:
            identity = curve_identity_key(curve)
            try:
                result = fitter(curve["energy"], curve["intensity"], config)
                if not result.diagnostics.valid:
                    failures[identity] = result.diagnostics.status
                else:
                    results[identity] = result
            except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                failures[identity] = str(exc)
        if not failures:
            self.applied_config, self.applied_results = config, results
            self.signal = "Corrected"
        return failures
