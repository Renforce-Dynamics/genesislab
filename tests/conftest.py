"""Pytest configuration: wire up ``genesislab`` and ``genesis_tasks`` source roots.

Keeps the suite runnable via ``pytest tests/`` from the repo root without requiring
``pip install -e .`` first — useful for quick sanity checks during development.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE_DIRS = [
    os.path.join(_REPO_ROOT, "source", "genesislab"),
    os.path.join(_REPO_ROOT, "source", "genesis_tasks"),
]

for _p in _SOURCE_DIRS:
    if _p not in sys.path and os.path.isdir(_p):
        sys.path.insert(0, _p)
