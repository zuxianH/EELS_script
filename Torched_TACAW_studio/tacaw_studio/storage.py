"""Atomic metadata and bounded Zarr-to-NumPy conversion."""
from __future__ import annotations

from itertools import product
import json
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import yaml


def atomic_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="."+path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as h:
            h.write(content)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_yaml(path: Path) -> dict:
    with path.open() as h:
        data = yaml.safe_load(h)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping.")
    return data


def atomic_yaml(path: Path, value: dict):
    atomic_text(path, yaml.safe_dump(value, sort_keys=False))


def atomic_json(path: Path, value: dict):
    atomic_text(path, json.dumps(value, indent=2))


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return ""


def inspect_array(path: Path) -> dict:
    import zarr
    array = zarr.open_array(str(path), mode="r")
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "chunks": list(array.chunks),
            "bytes": math.prod(array.shape)*np.dtype(array.dtype).itemsize,
            "chunk_bytes": math.prod(array.chunks)*np.dtype(array.dtype).itemsize}


def require_complete(config_path: Path) -> dict:
    c = read_yaml(config_path)
    status = c.get("computation_batches", {}).get("status_list", [])
    if not status or any(x != 2 for x in status):
        raise ValueError("Calculation is incomplete; NumPy conversion requires every batch to be finished.")
    return c


def copy_to_npy(array, output: Path, report=print):
    """Decode one storage chunk at a time; publish a complete file only."""
    if output.exists():
        raise FileExistsError(f"NumPy output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=output.name+".", suffix=".partial.npy", dir=output.parent)
    os.close(fd)
    partial = Path(name)
    try:
        mmap = np.lib.format.open_memmap(partial, mode="w+", shape=array.shape, dtype=array.dtype)
        mmap.flush()
        mmap._mmap.close()
        offsets = [range(0, size, chunk) for size, chunk in zip(array.shape, array.chunks)]
        total = math.prod(len(axis) for axis in offsets)
        for i, start in enumerate(product(*offsets), 1):
            selection = tuple(slice(x, min(x+c, n)) for x, c, n in zip(start, array.chunks, array.shape))
            block = array[selection]
            mmap = np.lib.format.open_memmap(partial, mode="r+")
            try:
                mmap[selection] = block
                mmap.flush()
            finally:
                mmap._mmap.close()
                del block
            report(f"Copied chunk {i}/{total}", flush=True)
        os.link(partial, output)
    finally:
        partial.unlink(missing_ok=True)


def array_revision(path: Path) -> str:
    """Fingerprint stored metadata/chunk attributes without decoding data."""
    digest = hashlib.sha256()
    for directory, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            p = Path(directory)/name
            stat = p.stat()
            digest.update(str(p.relative_to(path)).encode())
            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def convert_run(folder: Path):
    import zarr
    c = require_complete(folder/"scan_gpu_config.yaml")
    source = Path(c["storage"]["intensities_zarray"])
    output = source.with_suffix(".npy")
    info = inspect_array(source)
    revision = array_revision(source)
    marker = output.with_name(output.name+".conversion.json")
    print(json.dumps(info), flush=True)
    if output.exists():
        if not marker.is_file():
            raise FileExistsError(f"Existing NumPy file has no Studio conversion record: {output}. Choose a new output location rather than overwriting it.")
        recorded = json.loads(marker.read_text())
        stat = output.stat()
        if recorded.get("source") != str(source) or recorded.get("revision") != revision or recorded.get("output_bytes") != stat.st_size or recorded.get("output_mtime_ns") != stat.st_mtime_ns:
            raise ValueError("Existing NumPy output or source has changed since conversion. It will not be overwritten.")
        existing = np.load(output, mmap_mode="r", allow_pickle=False)
        try:
            if list(existing.shape) != info["shape"] or str(existing.dtype) != info["dtype"]:
                raise ValueError("Existing NumPy output has incompatible shape/dtype.")
        finally:
            existing._mmap.close()
        print(f"Keeping existing NumPy file: {output}", flush=True)
        return output
    import shutil
    if shutil.disk_usage(output.parent).free < info["bytes"] + 4096:
        raise ValueError(f"Insufficient free space for {human_bytes(info['bytes'])} NumPy output.")
    copy_to_npy(zarr.open_array(str(source), mode="r"), output)
    stat = output.stat()
    atomic_json(marker, {"source": str(source), "revision": revision,
                         "output_bytes": stat.st_size, "output_mtime_ns": stat.st_mtime_ns})
    return output
