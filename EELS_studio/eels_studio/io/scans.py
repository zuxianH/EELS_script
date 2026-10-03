"""Scan metadata and revision-checked opening; no intensity arrays are cached here."""
from dataclasses import dataclass
from pathlib import Path

from eels_studio.io.scan_sources import open_scan_reader, scan_revision


@dataclass(frozen=True)
class ScanInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    mtime_ns: int
    size: int
    revision: str = ""
    chunks: tuple[int, ...] | None = None

    @property
    def canonical_shape(self):
        if len(self.shape) == 3:
            energy, px, py = self.shape
            return (1, energy, 1, 1, px, py)
        return self.shape


def inspect_scan(path):
    path = Path(path).expanduser().resolve()
    scan = open_scan_reader(path)
    mtime_ns, size, revision = scan_revision(path)
    return ScanInfo(str(path), scan.shape, str(scan.dtype), mtime_ns, size,
                    revision, getattr(scan, "chunks", None))


def _open_scan(info):
    if scan_revision(info.path) != (info.mtime_ns, info.size, info.revision):
        raise ValueError("File changed on disk; refresh the file list")
    return open_scan_reader(info.path)
