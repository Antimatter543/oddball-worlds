"""Synthetic unit tests for the vetting (pytest spikes/mdwarf_stack/vetting/test_vet.py).

Cases: clean planet (tier A), EB with a secondary (secondary fails), single outlier event
(per-transit fails), orbit-edge systematic (systematics + orbit alias fail), rotation harmonic
(uniqueness fails). Default thresholds; these test the machinery, not the calibration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE))
from common import BIN_D
from inject import binned_model
from vet import attach_star, base_series, grade, make_ctx, measure

M = R = 0.3
SIG = 1e-3
STAR = {"gaia": {"target": {"source": "x", "G": 12.0}, "neighbours": []},
        "catalogue": {"objects": [], "sources": {"toi": "ok", "ctoi": "ok"}}}


def synth(seed=0, nsec=4, spacing=200.0, extra=None):
    rng = np.random.default_rng(seed)
    idx, sec = [], []
    for i in range(nsec):
        s0 = 1000.0 + i * spacing
        for a, b in ((s0, s0 + 13.0), (s0 + 13.7, s0 + 26.7)):  # two orbits, 0.7 d gap
            ii = np.arange(int(a / BIN_D), int(b / BIN_D))
            idx.append(ii)
            sec.append(np.full(len(ii), i + 1))
    idx = np.concatenate(idx)
    sec = np.concatenate(sec).astype(np.int16)
    t = (idx + 0.5) * BIN_D
    f = 1.0 + SIG * rng.standard_normal(len(idx))
    if extra is not None:
        f = f * extra(t, idx)
    lc = {"idx": idx, "flux": f.astype(np.float32), "err": np.full(len(idx), SIG, np.float32),
          "q": np.zeros(len(idx), np.int32), "sector": sec}
    sap = {"idx": idx, "sap": f.astype(np.float32), "bkg": np.full(len(idx), 0.01, np.float32),
           "sector": sec}
    return lc, sap


def vet_one(lc, sap, P, t0, k):
    base = base_series(lc, sap, {s: 1.0 for s in range(1, 10)})
    ctx = make_ctx(base, 1, M, R, "real", 0, {s: 1.0 for s in range(1, 10)}, STAR)
    sig = {"P": P, "t0": t0, "k": k, "mes": 0.0}
    st0 = measure(ctx, sig)
    sig["mes"] = st0["mes_vet"]
    st = attach_star(measure(ctx, sig), STAR)
    return st, grade(st)


def planet(P, t0, rp):
    return lambda t, idx: binned_model(idx, P=P, t0=t0, rp_re=rp, b=0.2, mass=M, radius=R)


def test_clean_planet_is_tier_a():
    P, t0 = 20.0, 1005.3
    lc, sap = synth(1, extra=planet(P, t0, 1.6))
    st, g = vet_one(lc, sap, P, t0, 12)
    assert st["n_transits"] >= 3
    for t in ("odd_even", "per_transit", "secondary", "model_compare", "uniqueness", "sap",
              "duration_density"):
        assert g["status"][t] == "pass", (t, g)
    assert g["tier"] == "A", g


def test_eb_secondary_fails():
    P, t0 = 20.0, 1005.3

    def eb(t, idx):
        ph = (t - t0) / P % 1.0
        f = np.ones(len(t))
        f[np.abs(ph - 0.0) * P < 0.05] -= 6e-3
        f[np.abs(ph - 1.0) * P < 0.05] -= 6e-3
        f[np.abs(ph - 0.5) * P < 0.05] -= 3e-3
        return f

    lc, sap = synth(2, extra=eb)
    st, g = vet_one(lc, sap, P, t0, 12)
    assert g["status"]["secondary"] == "fail", st["secondary"]
    assert g["tier"] == "C"


def test_single_outlier_event_fails():
    P, t0 = 20.0, 1005.3

    def one(t, idx):
        f = np.ones(len(t))
        f[np.abs(t - t0) < 0.05] -= 1.2e-2
        return f

    lc, sap = synth(3, extra=one)
    st, g = vet_one(lc, sap, P, t0, 12)
    assert g["status"]["per_transit"] == "fail", st["per_transit"]
    assert g["tier"] == "C"


def test_orbit_edge_systematic_fails():
    spacing = 13.7 * 15  # every orbit start lines up with P = 13.7 d

    def ramp(t, idx):
        f = np.ones(len(t))
        for i in range(4):
            for a in (1000.0 + i * spacing, 1000.0 + i * spacing + 13.7):
                m = (t >= a) & (t < a + 0.7)
                f[m] -= 4e-3 * (1 - (t[m] - a) / 0.7)
        return f

    lc, sap = synth(4, spacing=spacing, extra=ramp)
    st, g = vet_one(lc, sap, 13.7, 1000.0 + 0.35, 6)
    assert g["status"]["systematics_window"] == "fail", st["systematics_window"]
    assert g["status"]["orbit_alias"] == "fail", st["orbit_alias"]
    assert g["tier"] == "C"


def test_rotation_harmonic_not_unique():
    prot, t0 = 2.0, 1005.3

    def spots(t, idx):
        f = 1 + 0.01 * np.sin(2 * np.pi * t / prot)
        ph = ((t - t0) / prot + 0.5) % 1.0 - 0.5
        f = f - 3e-3 * (np.abs(ph * prot) < 0.05)  # a sharp spot feature every rotation
        return f

    lc, sap = synth(5, extra=spots)
    st, g = vet_one(lc, sap, 10 * prot, t0, 12)
    assert g["status"]["uniqueness"] == "fail", st["uniqueness"]
    assert g["tier"] == "C"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
