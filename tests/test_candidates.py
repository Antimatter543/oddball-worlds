import numpy as np
import pandas as pd

from oddball.score import candidates as K


def test_rank_desc_ties_and_nan():
    s = np.array([0.9, 0.99, 0.99, np.nan, -1.0])
    tie = np.array([0.0, 0.5, 0.7, 0.0, 0.0])
    tic = np.array([5, 4, 3, 2, 1])
    r = K.rank_desc(s, tie, tic)
    # 0.99 (tie 0.7) first, 0.99 (tie 0.5), 0.9, vetoed -1, NaN last
    assert r.tolist() == [3, 2, 1, 5, 4]


def test_classical_eb_veto_and_downweight():
    rng = np.random.default_rng(0)
    n = 400
    c = rng.uniform(size=n)
    strat = np.zeros(n, int)
    eb = np.zeros(n)
    top = int(np.argmax(c))
    eb[top] = 1.0  # a perfect EB at the top is pushed down
    flag = np.zeros(n, bool)
    flag[int(np.argsort(c)[-2])] = True  # the runner-up is vetoed
    sc, raw = K.classical_eb(c, eb, strat, flag, alpha=0.9)
    assert sc[flag][0] == -1.0 and raw[flag][0] > 0.99
    assert raw[top] < 0.9
    # with alpha 0 the ranking is the classical percentile (re-percentiled)
    sc0, _ = K.classical_eb(c, eb, strat, np.zeros(n, bool), alpha=0.0)
    assert np.array_equal(np.argsort(sc0), np.argsort(c))


def test_sector_candidates_union_and_merge():
    t = pd.DataFrame({"tic": np.arange(10), "cls_score": np.linspace(0, 1, 10),
                      "cls_raw": np.linspace(0, 1, 10), "ens_score": np.linspace(1, 0, 10),
                      "ens_raw": np.linspace(1, 0, 10)})  # fmt: skip
    c = K.sector_candidates(t, n=2)
    assert sorted(c.tic.tolist()) == [0, 1, 8, 9]
    a = c.assign(sector=32)
    b = K.sector_candidates(t.assign(cls_score=t.ens_score, cls_raw=t.ens_raw), n=1)
    b = b.assign(sector=33)
    w = K.merge_sectors(pd.concat([a, b]), {32: set(range(10)), 33: set(range(5)), 55: {99}})
    row0 = w.set_index("tic").loc[0]
    assert row0.multi and row0.n_sectors_cand == 2 and row0.sectors_cand == "32 33"
    assert row0.n_sectors_obs == 2 and row0.best_rank == 1
    assert not w.set_index("tic").loc[9].multi
    assert {"cls_rank_s0032", "ens_rank_s0033"} <= set(w.columns)


def test_ccd_excess_flags_a_hot_ccd():
    stars = pd.DataFrame({"camera": np.repeat([1, 2], 500), "ccd": 1})
    cand = pd.DataFrame({"camera": [1] * 45 + [2] * 5, "ccd": 1})
    t = K.ccd_excess(cand, stars).set_index("camera")
    assert t.loc[1, "ratio"] == 1.8 and t.loc[1, "p_excess"] < 1e-3
    assert t.loc[2, "p_excess"] > 0.5


def test_crowded_time_bins_flags_a_spacecraft_wide_spike():
    rng = np.random.default_rng(0)
    n = 500
    t = rng.uniform(2000.0, 2025.0, n)  # ~10 events per 0.5-d bin
    t[:30] = 2010.1 + rng.uniform(0, 0.3, 30)  # 30 extra events in one bin, all cameras
    t[30] = np.nan  # no residual event
    c = pd.DataFrame({"tic": np.arange(n), "cmv3_res_t": t, "camera": rng.integers(1, 5, n)})
    mask, bins = K.crowded_time_bins(c)
    assert list(bins.tbin) == [4020]
    assert mask[:30].all() and not mask[30]
    assert mask.sum() == bins.n.iloc[0]
    # a uniform list has no crowded bin
    c2 = c.assign(cmv3_res_t=rng.uniform(2000.0, 2025.0, n))
    m2, b2 = K.crowded_time_bins(c2)
    assert len(b2) == 0 and not m2.any()
