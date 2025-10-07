"""
Allow `python -m scripts.*` invocations to resolve to the project modules.

Some environments (notably ROS installs) ship a top-level `scripts` package that
can shadow our project modules. By explicitly extending this package's search
path to include `source/scripts`, we ensure the expected modules load first.
"""

import sys
from pathlib import Path
from pkgutil import extend_path

_package_dir = Path(__file__).resolve().parent
_source_scripts = _package_dir.parent / "source" / "scripts"
_source_root = _package_dir.parent / "source"

__path__ = extend_path(__path__, __name__)

if _source_scripts.is_dir():
    source_path = str(_source_scripts)
    if source_path not in __path__:
        __path__.append(source_path)

if _source_root.is_dir():
    source_root_path = str(_source_root)
    if source_root_path not in sys.path:
        sys.path.insert(0, source_root_path)
