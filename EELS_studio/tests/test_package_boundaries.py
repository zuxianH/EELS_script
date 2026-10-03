"""The scientific and numerical I/O package must work without importing the UI."""
from pathlib import Path
import subprocess
import sys


def test_core_and_io_import_without_streamlit():
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib
import pkgutil
import sys
import eels_studio.core
import eels_studio.io

for package in (eels_studio.core, eels_studio.io):
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        importlib.import_module(module.name)
assert not any(name == 'streamlit' or name.startswith('streamlit.') for name in sys.modules)
assert not any(name == 'eels_studio.ui' or name.startswith('eels_studio.ui.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=root,
                   check=True, capture_output=True, text=True, timeout=60)
