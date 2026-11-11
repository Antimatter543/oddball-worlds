"""Shared paths and constants for the all-sector M-dwarf stacking spike.

Data and outputs live in the MAIN checkout (git-ignored), by absolute path, so the worktree
needs no data/ of its own.
"""

from __future__ import annotations

import os
from pathlib import Path

MAIN = Path(os.environ.get("ODDBALL_MAIN", Path.home() / "Projects" / "oddball-worlds"))
DATA = MAIN / "data" / "planets_spike"
OUT = MAIN / "outputs" / "planets_spike"
LC_DIR = DATA / "lc"
CAT_DIR = DATA / "catalogs"

BIN_MIN = 10.0  # minutes; global bin grid k*w (oddball.tess.lightcurve)
BIN_D = BIN_MIN / 1440.0

# Generic client name: requests must not identify the user or project.
USER_AGENT = "python-requests/2 (scientific data client)"

for _d in (LC_DIR, CAT_DIR, OUT):
    _d.mkdir(parents=True, exist_ok=True)
