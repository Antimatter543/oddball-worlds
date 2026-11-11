"""FFA vs brute-force fold, and blind recovery of a synthetic transit across separated sectors."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BIN_D  # noqa: E402
from search import build_ses, ffa_search, fold_grid, search_star  # noqa: E402


def _fake(P=17.3, t0=1500.2, depth=0.004, dur=2.0 / 24, seed=1):
    rng = np.random.default_rng(seed)
    idx = []
    sec = []
    for s, start in enumerate([1400.0, 1700.0, 2300.0, 2327.0]):  # non-consecutive chunks
        i = np.arange(int(start / BIN_D), int((start + 25.0) / BIN_D))
        idx.append(i)
        sec.append(np.full(len(i), s + 1))
    idx = np.concatenate(idx)
    sec = np.concatenate(sec)
    t = (idx + 0.5) * BIN_D
    flux = 1 + rng.normal(0, 0.002, len(t))
    ph = (t - t0 + 0.5 * P) % P - 0.5 * P
    flux[np.abs(ph) < dur / 2] -= depth
    return {"idx": idx, "flux": flux, "sigma": np.full(len(t), 0.002), "sector": sec}


def test_ffa_matches_brute_on_integer_periods():
    pp = _fake()
    k, s = 12, 4
    dn = build_ses(pp, k, s, dense=True)
    binw = s * BIN_D
    P, mes, _, _ = ffa_search(dn, 15.0, 15.0 + 3 * binw, binw, 0.8)
    # rows with r=0 are exactly integer periods p*binw
    ints = np.isclose((P / binw) % 1.0, 0.0, atol=1e-9)
    t = dn["t_start"] + np.arange(len(dn["N"])) * binw
    v = dn["cnt"] > 0
    # an integer period appears twice: r=0 of p (exact fold) and r=m-1 of p-1 (cyclic-shift
    # approximation); at least one entry must equal the brute fold exactly
    assert ints.sum() >= 3
    for Pi in np.unique(np.round(P[ints], 9)):
        ref, _ = fold_grid(t[v] - dn["t_start"], dn["N"][v], dn["D"][v], np.array([Pi]), binw, 0.8)
        cands = mes[np.isclose(P, Pi, atol=1e-9)]
        assert np.min(np.abs(cands - ref[0])) < 1e-6 * max(1, ref[0]), (Pi, ref[0], cands)


def test_blind_recovery_synthetic():
    pp = _fake()
    sigs, stats = search_star(pp, 0.3, 0.3, max_sig=2)
    assert sigs, stats
    s0 = sigs[0]
    assert abs(s0.P - 17.3) < 0.01, s0
    assert s0.mes > 10
