"""One survey search: the frozen gate-2 method on one star's NPZ, in a worker process.

Exactly the gate-2 host / hold-out job path (`gate2/run2.run_task` with mode=real, no
injection): `search2.preprocess2(lc, rot)` once, then `search.search_star(pp, mass, radius,
max_sig, prange=run2.RANGES[range])` for the long (10-60 d) and short (1-10 d) ranges on the
same preprocessed light curve. Nothing found in one range is masked in the other.

  rot      from frozen.json ("harm"); never a default
  max_sig  6 per range, the gate-2 host and hold-out setting. The search is greedy and
           deterministic, so its first 3 signals are exactly what a max_sig=3 (real-control)
           run would return.
  fast     the frozen census rule: LS peak power >= fast_pow in >= fast_min_frac of sectors,
           from the same per-sector rot_sector call that harm makes (census.star_census).

The worker sets numba/BLAS to 1 thread before anything imports numba (as run2 does).
"""

from __future__ import annotations

import os

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))

MAX_SIG = 6


def load_npz(path: Path) -> tuple[dict, dict]:
    """Same as fetch.load, from an explicit path."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != "meta"}
        meta = json.loads(str(z["meta"]))
    return d, meta


def vmhwm_mb() -> float | None:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def search_one(tic: int, npz: str, mass: float, radius: float, cfg: dict) -> dict:
    from run2 import RANGES, _rot_short
    from search import search_star
    from search2 import preprocess2

    c0, w0 = time.process_time(), time.time()
    lc, meta = load_npz(Path(npz))
    if int(meta["tic"]) != tic:
        raise ValueError(f"NPZ tic {meta['tic']} != {tic}")
    pp, rinfo = preprocess2(lc, cfg["rot"])
    c1 = time.process_time()
    out = {"rot": _rot_short(rinfo), "cpu_pre": c1 - c0}
    pw = [v["power"] for v in (rinfo.get("sectors") or {}).values()]
    frac = float(np.mean(np.array(pw) >= cfg["fast_pow"])) if pw else 0.0
    out["rot"].update(frac_strong=frac, fast=bool(frac >= cfg["fast_min_frac"]),
                      per_sector={str(k): [v["P"], v["power"], v["applied"]]
                                  for k, v in (rinfo.get("sectors") or {}).items()})
    out["search"] = {}
    for rg in ("long", "short"):
        cs, ws = time.process_time(), time.time()
        sigs, stats = search_star(pp, mass, radius, MAX_SIG, prange=RANGES[rg])
        out["search"][rg] = {"prange": list(RANGES[rg]), "max_sig": MAX_SIG,
                             "signals": [s.as_dict() for s in sigs], "stats": stats,
                             "cpu_s": time.process_time() - cs, "wall_s": time.time() - ws}
    out["cpu_s"] = time.process_time() - c0
    out["wall_s"] = time.time() - w0
    out["n_bins_raw"] = int(len(lc["idx"]))
    out["worker_vmhwm_mb"] = vmhwm_mb()
    out["worker_pid"] = os.getpid()
    return out
