"""Star-level data splits shared by every training, injection and evaluation stage.

The contract (see docs/ML_DESIGN.md §3, §6):

* Splits are by **star (TIC)**, never by row, so a star lands in the same split in every sector.
* ``train`` (80%): self-supervised training only.
* ``val`` (10%): early stopping, and the **validation injections** used to tune thresholds and
  hyperparameters.
* ``test`` (10%): the **test injections** and final reported metrics. Nothing is tuned on it.
* ``known``: every TIC in ``docs/known_positives.csv``. Never trained on, never tuned on;
  scored blind for the gate. Overrides the hash split.

The hash is stable across machines and Python versions (no ``hash()``; it is salted per process).
"""

from __future__ import annotations

import csv
import hashlib
from functools import lru_cache

import numpy as np

from oddball.common.paths import REPO_ROOT

SPLITS = ("train", "val", "test", "known")
KNOWN_POSITIVES_CSV = REPO_ROOT / "docs" / "known_positives.csv"
_SALT = b"oddball-worlds-split-v1"


def _bucket(tic: int) -> int:
    """Stable 0-99 bucket for a TIC id."""
    h = hashlib.blake2b(str(int(tic)).encode(), digest_size=8, key=_SALT).digest()
    return int.from_bytes(h, "little") % 100


@lru_cache(maxsize=1)
def known_positive_tics() -> frozenset[int]:
    """TICs listed in docs/known_positives.csv (empty set if the file is missing)."""
    if not KNOWN_POSITIVES_CSV.exists():
        return frozenset()
    with KNOWN_POSITIVES_CSV.open(newline="") as f:
        return frozenset(int(r["tic"]) for r in csv.DictReader(f) if r.get("tic", "").strip())


def split_of(tic: int) -> str:
    """Return the split name for one TIC."""
    if int(tic) in known_positive_tics():
        return "known"
    b = _bucket(tic)
    if b < 80:
        return "train"
    if b < 90:
        return "val"
    return "test"


def split_array(tics) -> np.ndarray:
    """Vectorised :func:`split_of` over an iterable of TICs (returns an array of split names)."""
    return np.array([split_of(t) for t in tics], dtype=object)
