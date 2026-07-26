"""Robust motion capture from IMUs in everyday consumer devices.

The upstream research code historically used top-level imports such as
``from config import paths``.  Keep the package directory importable during the
transition so installed-package use and legacy scripts resolve the same code.
"""

import sys
from pathlib import Path


_PACKAGE_DIR = str(Path(__file__).resolve().parent)
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

__version__ = "0.2.0"
