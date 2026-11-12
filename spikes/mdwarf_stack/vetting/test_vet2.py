"""Synthetic unit tests for vetting v2 (pytest spikes/mdwarf_stack/vetting/test_vet2.py).

Machinery only (default v2 thresholds, v1 frozen thresholds for kept tests): clean planet is
A, a dip at a gap edge trips gap_edge, local red noise trips red_noise, the low-margin strike
fires only in the few-transit class and uses the selecting threshold, the revised pixel rule
needs 3 strong epochs, A+ needs an informative on-target pixel result, and the validation job
maker refuses real unaltered jobs.
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
from test_vet import STAR, M, R, planet, synth
from vet import attach_star, base_series, make_ctx, measure
from vet2 import NEW_DEFAULT, V1_KEPT, gap_edge, grade2, in_class_f, measure2, pixel2

V1_THR = {"odd_even.z": (3.0, 4.38), "per_transit.dom": (0.65, None),
          "per_transit.minses_neg": (1.11, 2.49), "duration_density.ratio": (1.38, 2.0),
          "secondary.excess": (1.0, 2.0), "model_compare.neg_dbic": (-6.0, 0.0),
          "uniqueness.neg_margin": (-2.0, -1.0), "sap.z_neg": (3.39, 6.72),
          "sap.bkg_stat": (4.0, None), "systematics_window.share": (0.5, 0.8),
          "orbit_alias.neg_spread": (-0.5, None), "rotation_residual.neg_logp": (1.69897, None)}
THR = {k: V1_THR[k] for k in V1_KEPT} | NEW_DEFAULT
TAU = 8.054


def vet2_one(lc, sap, P, t0, k, mode="real"):
    crowd = {s: 1.0 for s in range(1, 10)}
    base = base_series(lc, sap, crowd)
    ctx = make_ctx(base, 1, M, R, mode, 0, crowd, STAR)
    sig = {"P": P, "t0": t0, "k": k, "mes": 0.0}
    sig["mes"] = measure(ctx, sig)["mes_vet"]
    st = measure2(ctx, sig)
    if mode == "real":
        attach_star(st, STAR)
    return st


def test_thresholds_cover_exactly_the_v2_tests():
    assert set(THR) == set(V1_KEPT) | set(NEW_DEFAULT)
    assert "uniqueness.p_scr" not in THR and "contamination.need_neg" not in THR


def test_clean_planet_is_tier_a_and_new_tests_pass():
    P, t0 = 20.0, 1005.3
    lc, sap = synth(1, extra=planet(P, t0, 1.6))
    st = vet2_one(lc, sap, P, t0, 12)
    g = grade2(st, THR, TAU)
    assert st["v2"]["gap_edge"]["share"] < 0.5
    assert -st["v2"]["red_noise"]["neg_ratio"] > 0.8
    for t in ("gap_edge", "red_noise", "low_margin"):
        assert g["status"][t] == "pass", (t, g)
    assert g["tier"] == "A", g
    assert "contamination" not in g["status"] or g["status"]["contamination"] == "n/a"


def test_gap_edge_share_geometry():
    idx = np.concatenate([np.arange(0, 1000), np.arange(1010, 2000)])  # 100-min gap at 1000
    Dd = 6 * BIN_D
    c_near = (1000 + 3) * BIN_D  # window ends 30 min after the gap starts: inside it
    c_far = 1500 * BIN_D
    r = gap_edge(idx, np.array([c_near, c_far]), Dd, np.array([4.0, 1.0]))
    assert r["n_flagged"] == 1
    assert r["share"] == pytest.approx(16 / 17)
    c_70 = 1000 * BIN_D - 70 / 1440 - Dd / 2  # window ends 70 min before the gap
    r2 = gap_edge(idx, np.array([c_70, c_far]), Dd, np.array([4.0, 1.0]))
    assert r2["n_flagged"] == 0, r2
    c_50 = 1000 * BIN_D - 50 / 1440 - Dd / 2  # 50 min before: flagged
    assert gap_edge(idx, np.array([c_50]), Dd, np.array([4.0]))["n_flagged"] == 1


def test_dip_at_gap_edges_trips_gap_edge():
    # dips at the last kept data before every orbit gap (preprocessing trims 0.25 d per edge)
    lc, sap = synth(6)
    t = (lc["idx"] + 0.5) * BIN_D
    seg_end = t[np.flatnonzero(np.diff(t) > 0.3)]
    f = lc["flux"].astype(float)
    for e in seg_end:
        f[(t > e - 0.33) & (t <= e - 0.24)] -= 4e-3
    lc["flux"] = f.astype(np.float32)
    sap["sap"] = lc["flux"]
    st = vet2_one(lc, sap, 200.0, float(seg_end[0]) - 0.28, 6)
    assert st["v2"]["gap_edge"]["share"] > 0.8, st["v2"]["gap_edge"]
    g = grade2(st, THR, TAU)
    assert g["status"]["gap_edge"] == "fail"
    assert g["tier"] == "C"


def test_red_noise_ratio_drops_with_local_noise():
    P, t0 = 20.0, 1005.3
    rng = np.random.default_rng(9)

    def noisy(t, idx):
        f = planet(P, t0, 1.6)(t, idx)
        ph = (t - t0 + P / 2) % P - P / 2
        near = np.abs(ph) < 1.5
        # strong red noise (0.5 h correlation) only near the epochs
        red = np.convolve(rng.standard_normal(len(t)), np.ones(6) / 6, mode="same") * 6e-3
        return f * (1 + np.where(near, red, 0.0))

    lc, sap = synth(7, extra=noisy)
    st = vet2_one(lc, sap, P, t0, 12)
    assert -st["v2"]["red_noise"]["neg_ratio"] < 0.6, st["v2"]["red_noise"]
    assert grade2(st, THR, TAU)["status"]["red_noise"] == "fail"


def _stats(n_tr, P, mes, mode="inv"):
    return {"P": P, "t0": 0.0, "k": 12, "dur_h": 2.0, "mes": mes, "mode": mode,
            "n_transits": n_tr, "v2": {"gap_edge": {"share": 0.0},
                                        "red_noise": {"neg_ratio": -1.0}}}


def test_low_margin_class_and_selecting_threshold():
    assert in_class_f(2, 20.0) and in_class_f(3, 15.0) and in_class_f(4, 45.0)
    assert not in_class_f(4, 39.0) and not in_class_f(5, 50.0)
    thr = {"low_margin.neg_excess": (-1.0, None)}
    g = grade2(_stats(2, 50.0, 8.5), thr, TAU)  # 0.45 above tau: strike + duo strike -> C
    assert g["status"]["low_margin"] == "soft" and g["tier"] == "C"
    g = grade2(_stats(2, 50.0, 9.2), thr, TAU)  # 1.15 above: duo strike only -> B
    assert g["status"]["low_margin"] == "pass" and g["tier"] == "B"
    g = grade2(_stats(3, 50.0, 7.5), thr, 7.0)  # proxy population: margin above 7.0
    assert g["status"]["low_margin"] == "soft" and g["tier"] == "B"
    g = grade2(_stats(6, 50.0, 7.5), thr, 7.0)  # not in the class
    assert g["status"]["low_margin"] == "pass" and g["tier"] == "A"


def test_pixel_rule_needs_three_strong_epochs_and_a_plus():
    one = {"S_t": 5.0, "S_nb": 0.5, "n_strong": 1, "n_off": 1}
    assert pixel2(one)[0] == "pass"
    three = {"S_t": 5.0, "S_nb": 0.5, "n_strong": 3, "n_off": 2}
    assert pixel2(three)[0] == "soft"
    nb = {"S_t": 1.0, "S_nb": 6.0, "n_strong": 2, "n_off": 2}
    assert pixel2(nb)[0] == "fail"
    assert pixel2({"S_t": 1.0, "S_nb": float("nan"), "n_strong": 0, "n_off": 0})[0] == "n/a"
    st = _stats(6, 20.0, 12.0, mode="real")
    st["star"] = attach_star({"depth": 1e-3}, STAR)["star"]
    st["pixel"] = {"S_t": 6.0, "S_nb": 0.5, "n_strong": 3, "n_off": 0}
    thr = {"low_margin.neg_excess": (-1.0, None)}
    assert grade2(st, thr, TAU)["tier"] == "A+"
    st["pixel"] = {"S_t": 6.0, "S_nb": 0.5, "n_strong": 2, "n_off": 0}
    assert grade2(st, thr, TAU)["tier"] == "A"


def test_validation_jobs_refuse_real_unaltered():
    from select_val2 import check_jobs

    check_jobs([{"id": "a", "mode": "inv"}, {"id": "b", "mode": "real", "inj": {"P": 20}}])
    with pytest.raises(SystemExit):
        check_jobs([{"id": "c", "mode": "real"}])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
