"""Filesystem locations. All data lives under the data root, which git ignores."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """`$ODDBALL_DATA` if set, else `<repo>/data`."""
    env = os.environ.get("ODDBALL_DATA")
    return Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


def outputs_root() -> Path:
    return REPO_ROOT / "outputs"


def sector_dir(sector: int, product: str = "tess-spoc") -> Path:
    return data_root() / product / f"s{sector:04d}"
