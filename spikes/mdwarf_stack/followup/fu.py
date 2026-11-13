"""Follow-up of the Track C leftover signals (docs/spikes/mdwarf_followup.md).

Shared pieces:
  * the five signals and their ephemerides (weighted linear fit to the survey's detected epochs,
    per-epoch timing error sigma_i = max(D/4, D/max(SES_i, 1)), covariance inflated by the reduced
    chi^2 when it exceeds 1), propagated to every epoch in a sector where the star was on silicon;
  * the pre-registered single-epoch statistic: on a light curve run through the frozen gate-2
    preprocessing (search2.preprocess2, rot from frozen.json), a box of the signal's duration D
    centred at a trial time; depth = weighted mean deficit; SES = depth / (sigma_white * beta),
    beta = robust scatter of white-noise box SES over every other position of the same sector
    (the search's own per-sector red-noise rescaling, re-measured on this light curve);
  * box injection into the RAW normalised flux (before preprocessing), for the
    "would we have seen it" test.

Nothing here changes the survey search or the v2 vetting.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE.parent / "survey"))
sys.path.insert(0, str(ROOT / "src"))

from common import BIN_D, DATA, OUT  # noqa: E402

FU_OUT = OUT / "followup"
FU_DATA = DATA / "followup"  # downloaded light curves / cutouts (deleted at the end)
FU_OUT.mkdir(parents=True, exist_ok=True)
FU_DATA.mkdir(parents=True, exist_ok=True)

KEYS = ["survey-2102868-long#0", "survey-234295610-long#2", "survey-146636926-long#1",
        "survey-141816993-long#0", "survey-57830249-long#2"]
TICS = [int(k.split("-")[1]) for k in KEYS]

NSIG_WIN = 3.0          # window half-width = NSIG_WIN * sigma_pred (centre of the dip)
SES_SUPPORT_MIN = 3.0   # minimum SES for "supported" (raised to the random-window 95th pct)
SES_REC = 3.0           # an injected dip counts as recovered at SES >= this at its position
SES_EXP_REFUTE = 3.0    # refutation needs expected SES >= this
COVER_MIN = 0.5         # a box position counts as covered if >= 50% of its bins have data
WIN_COVER_MIN = 0.8     # refutation needs >= 80% of window positions covered


# --------------------------------------------------------------------------------------------
# signals and ephemerides


def load_signals() -> dict:
    rows = {}
    for line in (OUT / "survey" / "vet" / "graded.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["key"] in KEYS:
            rows[r["key"]] = r
    return rows


def fit_ephem(s: dict) -> dict:
    """Weighted linear ephemeris from the detected epochs. Returns T0 (at the reference epoch,
    the weighted-mean epoch number), P, cov, sigmas and residuals."""
    P0, D = s["P"], s["dur_h"] / 24
    t = np.asarray(s["epochs"], float)
    ses = np.asarray(s["epoch_ses"], float)
    n = np.round((t - t[0]) / P0)
    sig = np.maximum(D / 4, D / np.maximum(ses, 1.0))
    w = 1 / sig**2
    A = np.vstack([np.ones_like(n), n]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    coef = cov @ (A.T @ (w * t))
    res = t - A @ coef
    dof = len(t) - 2
    chi2 = float(np.sum(w * res**2))
    infl = max(1.0, chi2 / dof) if dof > 0 else 1.0
    cov = cov * infl
    return {"T0": float(coef[0]), "P": float(coef[1]), "cov": cov.tolist(), "t_first": float(t[0]),
            "sig_epoch_h": (24 * sig).tolist(), "res_h": (24 * res).tolist(), "chi2": chi2,
            "dof": dof, "infl": infl, "D": D, "depth": s["depth"], "P_search": P0}


def predict(e: dict, n: int) -> tuple[float, float]:
    """(mid-time, 1-sigma) for epoch number n (counted from the first detected epoch)."""
    v = np.array([1.0, n])
    cov = np.asarray(e["cov"])
    return e["T0"] + n * e["P"], float(math.sqrt(v @ cov @ v))


def epochs_between(e: dict, t_lo: float, t_hi: float) -> list[int]:
    k0 = math.ceil((t_lo - e["T0"]) / e["P"])
    k1 = math.floor((t_hi - e["T0"]) / e["P"])
    return list(range(k0, k1 + 1))


def sector_windows(ra: float, dec: float) -> list[dict]:
    """Sectors on silicon: union of tess-point 0.9.5 (planned table) and TESScut's own sector
    list (which also returns near-edge sectors). Window = tess-point mid +- 14 d."""
    import requests
    import tess_stars2px as t2p

    from oddball.common.http import ANON_USER_AGENT

    out = t2p.tess_stars2px_function_entry(1, ra, dec)
    tp = {int(s) for s in np.atleast_1d(out[3])}
    tc, tc_status = set(), "ok"
    try:
        r = requests.get("https://mast.stsci.edu/tesscut/api/v0.1/sector",
                         params={"ra": ra, "dec": dec, "radius": "0m"},
                         headers={"User-Agent": ANON_USER_AGENT}, timeout=60)
        r.raise_for_status()
        tc = {int(x["sector"]) for x in r.json()["results"]}
    except Exception as ex:  # noqa: BLE001
        tc_status = f"failed: {ex!r}"[:120]
    sc = t2p.TESS_Spacecraft_Pointing_Data()
    mid = {int(s): float(m) - 2457000 for s, m in zip(sc.sectors, sc.midtimes)}
    secs = sorted(tp | tc)
    return [{"sector": s, "tesspoint": s in tp, "tesscut": s in tc, "tesscut_status": tc_status,
             "lo": mid[s] - 14.0, "hi": mid[s] + 14.0} for s in secs if s in mid]


# --------------------------------------------------------------------------------------------
# light curves and the single-epoch statistic


def frozen_rot() -> str:
    return json.loads((OUT / "gate2" / "frozen.json").read_text())["rot"]


def prep(lc: dict) -> dict:
    from search2 import preprocess2

    pp, _ = preprocess2(lc, rot=frozen_rot())
    return pp


def inject_box(lc: dict, tc: float, D: float, depth: float) -> dict:
    """Multiply raw normalised flux by a box dip (bin-overlap weighted)."""
    t0 = lc["idx"] * BIN_D
    t1 = t0 + BIN_D
    ov = np.clip(np.minimum(t1, tc + D / 2) - np.maximum(t0, tc - D / 2), 0, None) / BIN_D
    out = dict(lc)
    out["flux"] = lc["flux"].astype(np.float64) * (1 - depth * ov)
    return out


class Meter:
    """Box statistic of duration D on a preprocessed light curve."""

    def __init__(self, pp: dict, D: float, exclude: list[float] = ()):
        self.t = (pp["idx"] + 0.5) * BIN_D
        self.f = pp["flux"]
        self.w = 1 / pp["sigma"] ** 2
        self.sec = pp["sector"]
        self.D = D
        self.k = max(1, int(round(D / BIN_D)))
        self.beta = {}
        for s in np.unique(self.sec):
            m = self.sec == s
            ts = self.t[m]
            grid = np.arange(ts.min() + D / 2, ts.max() - D / 2, D / 2)
            ok = np.ones(len(grid), bool)
            for x in exclude:
                ok &= np.abs(grid - x) > 1.5 * D
            vals = [v for v in (self._white(g) for g in grid[ok]) if v is not None]
            vals = np.array([v[0] / v[1] for v in vals])
            if len(vals) >= 20:
                r = 1.4826 * np.median(np.abs(vals - np.median(vals)))
                self.beta[int(s)] = max(1.0, float(r))
            else:
                self.beta[int(s)] = float("nan")

    def _white(self, tc):
        i0, i1 = np.searchsorted(self.t, [tc - self.D / 2, tc + self.D / 2])
        n = i1 - i0
        if n < COVER_MIN * self.k:
            return None
        w = self.w[i0:i1]
        d = float(np.sum(w * (1 - self.f[i0:i1])) / np.sum(w))
        return d, float(1 / math.sqrt(np.sum(w))), n, int(np.bincount(self.sec[i0:i1]).argmax())

    def at(self, tc: float) -> dict | None:
        v = self._white(tc)
        if v is None:
            return None
        d, sw, n, s = v
        b = self.beta.get(s, float("nan"))
        sb = sw * b
        return {"t": tc, "depth": d, "sigma": sb, "ses": d / sb, "cover": n / self.k, "sector": s,
                "beta": b}

    def window(self, tc: float, half: float) -> list[dict]:
        grid = tc + np.arange(-half, half + 1e-9, BIN_D) if half > 0 else np.array([tc])
        return [m for m in (self.at(g) for g in grid)]


def random_windows(pp: dict, D: float, half: float, avoid: list[float], n: int = 400,
                   seed: int = 0) -> np.ndarray:
    """Max SES over windows of the same half-width at random covered positions away from every
    predicted epoch (+-2 days), for the empirical false-support rate."""
    rng = np.random.default_rng(seed)
    met = Meter(pp, D, exclude=avoid)
    t = met.t
    out = []
    tries = 0
    while len(out) < n and tries < 50 * n:
        tries += 1
        u = float(rng.choice(t))
        if any(abs(u - a) < 2.0 for a in avoid):
            continue
        ms = [m for m in met.window(u, half) if m is not None and np.isfinite(m["ses"])]
        if len(ms) < 0.8 * max(1, len(met.window(u, half))):
            continue
        out.append(max(m["ses"] for m in ms))
    return np.array(out)
