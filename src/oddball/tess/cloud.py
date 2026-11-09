"""Pure helpers for running the sector ingest as a Kaggle kernel (no network, no kaggle import).

The driver is ``scripts/kaggle_ingest.py``; the kernel body is ``scripts/kaggle/ingest_kernel.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

CONFIG_MARKER = "CONFIG = {}  # @@CONFIG@@"
KERNEL_FILE = "ingest_kernel.py"


def kernel_title(sector: int, limit: int | None = None) -> str:
    """Title whose Kaggle slug is exactly :func:`kernel_slug` (lowercase words, single spaces)."""
    return f"oddball ingest s{sector:04d}" + (f" l{limit}" if limit else "")


def kernel_slug(sector: int, limit: int | None = None) -> str:
    return kernel_title(sector, limit).replace(" ", "-")


def kernel_metadata(username: str, sector: int, limit: int | None = None) -> dict:
    """``kernel-metadata.json`` for a private, CPU-only, internet-enabled script kernel."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username or ""):
        raise ValueError(f"bad Kaggle username {username!r}")
    return {
        "id": f"{username}/{kernel_slug(sector, limit)}",
        "title": kernel_title(sector, limit),
        "code_file": KERNEL_FILE,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def kernel_config(
    sector: int,
    sha: str,
    kernel: str,
    pushed_utc: str,
    limit: int | None = None,
    include_tics: list[int] | None = None,
    workers: int = 32,
    shard_size: int = 10_000,
    deadline_hours: float = 11.5,
    retry_passes: int = 2,
) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError(f"pin a full 40-hex commit SHA, got {sha!r}")
    if not 0 < deadline_hours < 12:
        raise ValueError("deadline_hours must be inside Kaggle's 12 h session limit")
    return {
        "sector": int(sector),
        "sha": sha,
        "kernel": kernel,
        "pushed_utc": pushed_utc,  # also a nonce: the driver checks it comes back in provenance
        "limit": int(limit) if limit else None,
        "include_tics": sorted(int(t) for t in include_tics or []),
        "workers": int(workers),
        "shard_size": int(shard_size),
        "deadline_hours": float(deadline_hours),
        "retry_passes": int(retry_passes),
    }


def render_kernel(template: str, config: dict) -> str:
    """Substitute the CONFIG line of the kernel template with a JSON literal of ``config``."""
    if template.count(CONFIG_MARKER) != 1:
        raise ValueError("kernel template must contain the CONFIG marker line exactly once")
    # JSON with true/false/null is not Python; repr() of the json-roundtripped dict is.
    literal = repr(json.loads(json.dumps(config)))
    code = template.replace(CONFIG_MARKER, f"CONFIG = {literal}")
    compile(code, KERNEL_FILE, "exec")
    return code


def output_relpath(file_name: str, sector: int) -> str | None:
    """Map a kernel output file name to a path relative to the local sector dir.

    Kernel outputs are named relative to ``/kaggle/working``, e.g.
    ``tess-spoc/s0033/shards/lc_00000.parquet`` -> ``shards/lc_00000.parquet``. Anything outside
    that sector dir, or any path that could escape it, returns None (not downloaded).
    """
    p = PurePosixPath(file_name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts:
        return None
    prefix = ("tess-spoc", f"s{sector:04d}")
    if p.parts[:2] != prefix or len(p.parts) < 3:
        return None
    return str(PurePosixPath(*p.parts[2:]))
