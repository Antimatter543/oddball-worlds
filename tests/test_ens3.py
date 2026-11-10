import numpy as np
import pandas as pd
import pytest

from oddball.score import ens3 as E3


def test_top1_symmetric_veto_and_split():
    real = pd.DataFrame({"split": ["val"] * 200 + ["test"] * 200, "stratum": 0})
    rs = np.r_[np.arange(200.0), np.arange(200.0)]
    rf = np.zeros(400, bool)
    q = pd.DataFrame({"split": ["val", "val", "val", "test"], "stratum": 0})
    qs = np.array([1000.0, 1000.0, 197.0, 1000.0])
    qf = np.array([False, True, False, False])
    hit = E3.top1(real, q, rs, qs, rf, qf, "val")
    assert hit.tolist() == [True, False, False]  # vetoed row never counts; 197 is below 99%
    # vetoing the top real stars lowers the cut, so a mid score gets in
    rf2 = rf.copy()
    rf2[190:200] = True
    assert E3.top1(real, q, rs, qs, rf2, qf, "val").tolist() == [True, False, True]


def _terms(**kw):
    t = {f"gen_{d}": 0.5 for d in E3.GEN_DEPTHS}
    t.update({f"yso_{d}": 0.4 for d in E3.YSO_DEPTHS})
    t.update({f"fpr_{k}": 0.1 for k in E3.INST_NULLS})
    t.update(fpr_null_eb=0.2, fpr_null_yso=0.3, fpr_v1_pooled=0.05)
    t.update(kw)
    return t


def test_objective_terms():
    o = E3.objective(_terms())
    assert o["J"] == pytest.approx(0.5 + 0.4 - 0.1 - 0.2)
    assert o["J_v2"] == pytest.approx(0.5 - 0.05)
    assert o["N_yso"] == 0.3  # reported, not in J
    assert E3.objective(_terms(fpr_null_yso=0.9))["J"] == o["J"]


def test_pick_simplest_within_tol_and_dusty_guard():
    sweep = [
        {"rule": "wmax", "weights": [1, 2, 2], "alpha": 0.0, "J": 0.500, "ok_dusty": True},
        {"rule": "wmax", "weights": [1, 1, 0.5], "alpha": 0.5, "J": 0.495, "ok_dusty": True},
        {"rule": "max", "weights": None, "alpha": 0.75, "J": 0.492, "ok_dusty": True},
        {"rule": "mean", "weights": None, "alpha": 0.0, "J": 0.600, "ok_dusty": False},
        {"rule": "fisher", "weights": None, "alpha": 0.0, "J": 0.499, "ok_dusty": True},
    ]
    p = E3.pick(sweep)
    assert p["rule"] == "max"  # mean is excluded by the guard; max is simplest in the band
    p = E3.pick(sweep, tol=0.006)
    assert p["weights"] == [1, 1, 0.5]  # closer to equal weights than 1:2:2
    assert E3.simplicity({"rule": "wmax", "weights": [1, 0.5, 2], "alpha": 0}) == (2, 2.0, 0.0)


def test_cluster_bootstrap_uses_clusters():
    hit = np.r_[np.ones(50), np.zeros(50)]
    m, lo, hi = E3.cluster_bootstrap(hit, np.r_[np.zeros(50), np.ones(50)], n_boot=500)
    assert m == 0.5 and lo == 0.0 and hi == 1.0  # two clusters: the interval is wide
    m, lo, hi = E3.cluster_bootstrap(hit, np.arange(100), n_boot=500)
    assert 0.35 < lo < 0.5 < hi < 0.65


def test_paired_delta_identical_candidates_is_zero():
    rng = np.random.default_rng(0)
    depth_bins = np.array(E3.YSO_DEPTHS, dtype=object)
    sets = {
        "v1": pd.DataFrame({"inj_id": [f"a{i}" for i in range(60)], "event_id": -1,
                            "is_null": [False] * 45 + [True] * 15,
                            "kind": ["x"] * 45 + ["null_ccd"] * 8 + ["null_dump"] * 7,
                            "depth_bin": pd.Categorical(np.resize(depth_bins[:3], 60))}),
        "n2": pd.DataFrame({"inj_id": [f"b{i}" for i in range(30)], "event_id": np.arange(30) // 10,
                            "is_null": True,
                            "kind": np.repeat(["null_asteroid", "null_step", "null_column"], 10),
                            "depth_bin": pd.Categorical([None] * 30)}),
        "eb1": pd.DataFrame({"inj_id": [f"c{i}" for i in range(20)], "event_id": -1,
                             "is_null": [True] * 10 + [False] * 10,
                             "kind": ["null_eb"] * 10 + ["dusty_tail"] * 10,
                             "depth_bin": pd.Categorical([None] * 20)}),
        "yso1": pd.DataFrame({"inj_id": [f"d{i}" for i in range(44)], "event_id": -1,
                              "host_tic": np.r_[np.arange(40) // 10, np.arange(4)], "is_null": [False] * 40 + [True] * 4,
                              "kind": ["aperiodic_dipper"] * 40 + ["null_yso"] * 4,
                              "depth_bin": pd.Categorical(np.resize(depth_bins, 44))}),
    }  # fmt: skip
    h = {n: rng.uniform(size=len(q)) < 0.4 for n, q in sets.items()}
    d0, lo, hi = E3.paired_delta_j(sets, h, h, n_boot=50)
    assert d0 == 0.0 and lo == 0.0 and hi == 0.0
    h2 = {n: v.copy() for n, v in h.items()}
    h2["yso1"][:40] = True  # a candidate that finds every young-host injection
    d0, lo, hi = E3.paired_delta_j(sets, h2, h, n_boot=50)
    assert d0 > 0 and lo > 0
    assert set(E3.clusters(sets["n2"], "n2")) == {"ev0", "ev1", "ev2"}
