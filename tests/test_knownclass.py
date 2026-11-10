import numpy as np
import pandas as pd
import pytest

from oddball.knownclass import labels as L
from oddball.knownclass.explain import (
    ExplainAway,
    WeightedECDF,
    choose_alpha,
    downweight,
    explained_share,
)
from oddball.knownclass.features import (
    close_gaps,
    dip_event_features,
    event_stats,
    periodogram_features,
)
from oddball.knownclass.model import (
    KnownClassModel,
    _knn_mean_dist,
    _softmax,
    cap_quiet,
    fit_temperature,
    lc_quiet,
    training_targets,
)
from oddball.score.classical import channel_features

BIN = 30.0 / 1440.0
ANOMALY_TARGETS = {"dipper", "yso", "boyajian", "exocomet", "disintegrating"}


# ---------------------------------------------------------------------------------------------
# labels


@pytest.mark.parametrize("t,cls", [
    ("EA", "EB"), ("EW", "EB"), ("EA|ROT", "EB"), ("ROT", "ROT"), ("BY", "ROT"),
    ("RRAB", "RR"), ("RRC", "RR"), ("DSCT|GDOR", "DSCT_GDOR"), ("HADS", "DSCT_GDOR"),
    ("SR", "LPV"), ("M", "LPV"), ("LB", "LPV"), ("DCEP", "CEP"), ("CWA", "CEP"),
    ("ELL", "ELL"), ("BCEP", "BCEP_SPB"),
    ("EW:", None), ("VAR", None), ("MISC", None), ("", None), ("UV", None),
])  # fmt: skip
def test_vsx_class(t, cls):
    assert L.vsx_class(t) == cls


@pytest.mark.parametrize("t", ["YSO", "UXOR", "INSA", "INST", "IN", "IS", "CTTS/ROT", "TTS",
                               "WTTS", "FUOR", "EXOR", "DYPER", "ROT+YSO", "INT:", "RCB",
                               "ISB", "I"])  # fmt: skip
def test_vsx_excluded_types_are_never_labels(t):
    assert L.vsx_excluded(t)
    assert L.vsx_class(t) is None


@pytest.mark.parametrize("t", ["EA", "ROT", "RRAB", "SR", "DSCT", "LB", "BY", "E", "ELL"])
def test_ordinary_types_not_excluded(t):
    assert not L.vsx_excluded(t)


def test_no_class_is_an_anomaly_target():
    assert not {c.lower() for c in L.CLASSES} & ANOMALY_TARGETS
    for names in L.VSX_CLASS.values():
        assert not any(L.vsx_excluded(n) for n in names)
    assert not set(L.GAIA_CLASSIFIER) & L.GAIA_EXCLUDE


def _cats():
    vsx = pd.DataFrame({
        "tic": [1, 2, 3, 3, 5, 6],
        "Name": ["a", "b", "c", "c2", "e", "f"],
        "Type": ["EA", "ROT", "RRAB", "UXOR", "EW", "SR"],
        "sep_arcsec": [1.0, 2.0, 1.0, 15.0, 1.0, 12.0],  # TIC 3: a YSO one pixel away
    })  # fmt: skip
    prsa = pd.DataFrame({"TIC": [2], "Per": [1.3], "Morph": [0.4]})
    gaia = pd.DataFrame({
        "tic": [4, 7, 8], "source_id": [10, 11, 12], "gmag": [10.5, 10.3, np.nan],
        "tmag": [10.0, 10.0, 10.0], "varflag": ["VARIABLE"] * 3, "sep_arcsec": [0.3, 0.2, 0.5],
        "best_class_name": ["DSCT|GDOR|SXPHE", "YSO", "LPV"], "best_class_score": [0.9, 0.9, 0.7],
    })  # fmt: skip
    return vsx, prsa, gaia


def test_build_labels_priority_and_exclusions():
    vsx, prsa, gaia = _cats()
    tics = np.arange(1, 10)
    lab = L.build_labels(tics, np.full(9, 10.0), vsx, prsa, gaia, {5: "known_positive",
                                                                  9: "injection_host"})
    lab = lab.set_index("tic")
    assert lab.loc[1, "label"] == "EB" and lab.loc[1, "label_source"] == "vsx"
    assert lab.loc[2, "label"] == "EB" and lab.loc[2, "label_source"] == "prsa2022"  # > vsx ROT
    assert lab.loc[2, "n_sources_disagree"] == 1
    assert lab.loc[3, "excluded"] and lab.loc[3, "label"] == ""  # blended YSO excludes
    assert lab.loc[4, "label"] == "DSCT_GDOR" and lab.loc[4, "label_source"] == "gaia_classifier"
    assert lab.loc[5, "excluded"] and lab.loc[5, "exclude_reason"] == "known_positive"
    assert lab.loc[5, "label"] == ""
    assert lab.loc[6, "label"] == ""  # SR 12" away: blended, flagged but not a label
    assert lab.loc[6, "flagged_any"] and not lab.loc[6, "excluded"]
    assert lab.loc[7, "excluded"] and lab.loc[7, "exclude_reason"] == "gaia_class"
    assert lab.loc[8, "label"] == "LPV"  # no photometry: position match only
    assert lab.loc[9, "excluded"] and not lab.loc[9, "flagged_any"]
    assert not lab.kostov_nn.any()


def test_build_labels_kostov_and_gaia_veb():
    vsx, prsa, gaia = _cats()
    kostov = {"table3": pd.DataFrame({"TIC": [4, 6], "Per": [2.0, 3.0]}),
              "table4": pd.DataFrame({"TIC": [3], "Per_TESS": [1.1]}),
              "table2": pd.DataFrame({"TIC": [1, 8, 9]})}  # fmt: skip
    veb = pd.DataFrame({"tic": [7, 8, 1], "Source": [1, 2, 3], "Freq": [0.5, 0.2, 1.0],
                        "ModelType": ["TWOGAUSSIANS"] * 3, "sep_arcsec": [0.5, 2.5, 0.1]})  # fmt: skip
    lab = L.build_labels(np.arange(1, 10), np.full(9, 10.0), vsx, prsa, gaia, {},
                         kostov=kostov, gaia_veb=veb).set_index("tic")  # fmt: skip
    assert lab.loc[4, "label_source"] == "kostov2025_table3"  # beats the Gaia classifier
    assert lab.loc[6, "label"] == "EB"  # vetted TESS EB beats a blended VSX SR
    assert lab.loc[3, "label"] == "" and lab.loc[3, "excluded"]  # exclusions still win
    assert lab.loc[7, "excluded"]  # Gaia YSO excludes even with a veb match
    assert lab.loc[8, "label"] == "LPV"  # veb match 2.5" away: not a label
    assert lab.loc[1, "label_source"] == "vsx"  # VSX outranks gaia_veb
    assert lab.loc[[1, 8, 9], "kostov_nn"].all() and not lab.loc[[2, 4], "kostov_nn"].any()
    assert lab.loc[9, "label"] == ""  # the NN flag alone never labels


# ---------------------------------------------------------------------------------------------
# training targets


def _feat(n, var_frac, dip, bump, inj=None, tics=None):
    return pd.DataFrame({"tic": tics if tics is not None else np.arange(n), "inj_id": inj,
                         "pdc_var_frac": var_frac, "pdc_dip_snr": dip, "pdc_bump_snr": bump})


def test_training_targets_gates():
    labels = pd.DataFrame({
        "tic": [0, 1, 2, 3, 4, 5],
        "label": ["EB", "EB", "", "", "", "ROT"],
        "excluded": [False, False, False, False, True, True],
        "flagged_any": [True, True, False, True, False, True],
    })  # fmt: skip
    f = _feat(6, [0.9, 0.05, 0.05, 0.05, 0.05, 0.9], [30, 4, 4, 4, 4, 30], [3, 3, 3, 3, 3, 3])
    y = training_targets(f, labels, other=True).tolist()
    # EB that varies; EB flat in this sector dropped; quiet; flagged-but-flat is not quiet but
    # "other"; excluded stars are never targets (not quiet, not other, label unused)
    assert y == ["EB", "", "quiet", "other", "", ""]
    assert training_targets(f, labels, other=False).tolist() == ["EB", "", "quiet", "", "", ""]
    f2 = f.assign(inj_id=["x"] * 6)
    assert set(training_targets(f2, labels)) == {""}


def test_lc_quiet_requires_no_dip():
    f = _feat(3, [0.05, 0.05, 0.5], [4, 9, 4], [4, 4, 4])
    assert lc_quiet(f).tolist() == [True, False, False]


def test_cap_quiet():
    y = pd.Series(["quiet"] * 100 + ["other"] * 50 + ["EB"] * 10 + ["RR"] * 3 + [""] * 5)
    c = cap_quiet(y, cap=2.0)
    assert (c == "quiet").sum() == 20 and (c == "other").sum() == 20
    assert (c == "EB").sum() == 10 and (c == "RR").sum() == 3


# ---------------------------------------------------------------------------------------------
# features


def _grid(n_rows=4, seed=0):
    rng = np.random.default_rng(seed)
    t = (104363 + np.arange(1245) + 0.5) * BIN
    f = 1 + 5e-4 * rng.standard_normal((n_rows, t.size))
    f[:, 560:640] = np.nan
    return t, f


def test_close_gaps_and_event_stats():
    m = np.zeros((2, 30), bool)
    m[0, [2, 3, 5, 6, 15, 16, 25]] = True  # gap of 1 at 4 -> merged; separate runs at 15, 25
    m[1, [2, 10, 18, 26]] = True  # four evenly spaced events
    c = close_gaps(m, 2)
    n, irr = event_stats(c)
    assert n.tolist() == [3, 4]
    assert irr[1] == pytest.approx(0.0)
    m2 = np.zeros((1, 40), bool)
    m2[0, [2, 10, 26, 34]] = True  # periodic with one event missing (in a gap)
    assert event_stats(m2)[1][0] == pytest.approx(0.0)
    m2[0, 13] = True  # an off-beat event
    assert event_stats(m2)[1][0] > 0.15
    assert c[0, 4] and not c[0, 10]


def test_periodogram_finds_sinusoid_period():
    t, f = _grid()
    f[0] *= 1 + 0.01 * np.sin(2 * np.pi * t / 1.7)
    f[1] *= 1 + 0.003 * np.sin(2 * np.pi * t / 0.15)  # delta Scuti-like, 6.7 c/d
    x = f / np.nanmedian(f, axis=1, keepdims=True) - 1
    o = periodogram_features(x, t)
    assert 10 ** o["pg_logP"][0] == pytest.approx(1.7, rel=0.03)
    assert o["pg_peak_frac"][0] > 0.8 and o["pg_r2_p"][0] > 0.9
    assert 10 ** o["pg_logP"][1] == pytest.approx(0.15, rel=0.03)
    assert o["pg_hf_frac"][1] > 0.8 and o["pg_hf_frac"][0] < 0.1
    assert o["pg_peak_frac"][2] < 0.2  # noise


def _eclipses(t, period, d1, d2, w=0.08):
    ph = np.mod((t - t[0] - 0.3) / period, 1.0)
    m = np.ones_like(t)
    m -= d1 * np.exp(-0.5 * (np.minimum(ph, 1 - ph) * period / (w / 2)) ** 2)
    m -= d2 * np.exp(-0.5 * ((ph - 0.5) * period / (w / 2)) ** 2)
    return m


def test_eb_vs_single_dip_event_features():
    t, f = _grid()
    f[0] *= _eclipses(t, 2.9, 0.2, 0.05)  # detached EB: unequal eclipses every 2.9 d
    f[1] *= 1 - 0.2 * np.exp(-0.5 * ((t - t[300]) / 0.04) ** 2)  # one isolated dip
    pdc = channel_features(f, np.full_like(f, 5e-4), t, BIN)
    ev = dip_event_features(pdc.resid, pdc.snr4, pdc.feats["dip_depth"])
    assert ev["ev_n_half"][0] >= 6 and ev["ev_irreg_half"][0] < 0.05
    assert ev["ev_n_half"][1] == 1 and np.isnan(ev["ev_irreg_half"][1])
    o = periodogram_features(pdc.x, t)
    assert o["pg_prof_skew"][0] < 0.2  # narrow eclipses: the median sits near the maximum


# ---------------------------------------------------------------------------------------------
# calibration + explain-away


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true = rng.normal(size=(20000, 4)) * 2.0
    y = np.array([rng.choice(4, p=p) for p in _softmax(true)])
    T = fit_temperature(true * 3.0, y)  # logits 3x overconfident
    assert T == pytest.approx(3.0, rel=0.1)


def test_weighted_ecdf_matches_unweighted():
    rng = np.random.default_rng(1)
    v = rng.random(1000)
    e = WeightedECDF().fit(v, np.ones((1000, 1)))
    q = np.array([0.0, 0.25, 0.5, 1.0])
    assert np.allclose(e.transform(q)[:, 0], [0, 0.25, 0.5, 1.0], atol=0.03)


def test_explain_away_ranks_quiet_dip_above_ordinary_eb():
    rng = np.random.default_rng(2)
    n = 20000
    # reference: 10% EBs with very high base percentiles, 90% quiet with uniform-ish ones
    is_eb = rng.random(n) < 0.1
    s = np.where(is_eb, 0.97 + 0.03 * rng.random(n), 0.97 * rng.random(n))
    p = np.stack([is_eb, ~is_eb], 1).astype(float)
    ea = ExplainAway(["EB", "quiet"]).fit(s, p)
    eb_typical = ea.ka(np.array([0.985]), np.array([[1.0, 0.0]]))[0]
    quiet_dip = ea.ka(np.array([0.97]), np.array([[0.0, 1.0]]))[0]
    assert eb_typical == pytest.approx(0.5, abs=0.05)
    assert quiet_dip > 0.95
    # alpha = 0 is the base score
    assert ea.score(np.array([0.3]), np.array([[1.0, 0.0]]), 0.0)[0] == pytest.approx(0.3)


def test_other_class_explains_nothing():
    rng = np.random.default_rng(3)
    s = rng.random(5000)
    p = np.tile([0.5, 0.5, 0.0], (5000, 1))
    ea = ExplainAway(["EB", "other", "RR"]).fit(s, p)  # RR: no reference mass -> identity
    q = np.array([0.2, 0.9])
    assert np.allclose(ea.ka(q, np.array([[0, 1.0, 0], [0, 0, 1.0]])), q)
    assert ea.identity.tolist() == [False, True, True]


def test_choose_alpha_prefers_smaller_on_ties():
    assert choose_alpha({0.0: 0.1, 0.5: 0.3, 1.0: 0.3}) == 0.5


def test_downweight_and_explained_share():
    s = np.array([0.999, 0.999, 0.5])
    e = np.array([0.0, 1.0, 0.5])
    out = downweight(s, e, 1.0)
    assert out[0] == pytest.approx(0.999) and out[1] == pytest.approx(0.0)
    assert 0.0 < out[2] < 0.5
    assert np.allclose(downweight(s, e, 0.0), s)
    kc = pd.DataFrame({"kc_p_EB": [0.9, 0.9, 0.2], "kc_t_EB": [0.5, 0.02, 0.5],
                       "kc_p_other": [0.1, 0.1, 0.8]})  # no kc_t_other: other explains nothing
    assert np.allclose(explained_share(kc, 0.1), [0.9, 0.18, 0.2])


def test_knn_mean_dist_skips_own_tic():
    ref = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    q = np.array([[0.0, 0.0]])
    assert _knn_mean_dist(q, ref, 1)[0] == pytest.approx(0.0)
    d = _knn_mean_dist(q, ref, 1, q_tic=np.array([7]), r_tic=np.array([7, 8, 9]))
    assert d[0] == pytest.approx(10.0)


def test_classifier_and_typicality_on_toy_classes():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    n = 600
    y = np.array(["EB"] * n + ["ROT"] * n + ["other"] * n)
    centre = {"EB": (3.0, 0.0), "ROT": (0.0, 3.0), "other": (0.0, 0.0)}
    xy = np.array([centre[c] for c in y]) + rng.normal(scale=0.5, size=(3 * n, 2))
    X = pd.DataFrame({"pg_a": xy[:, 0], "pg_b": xy[:, 1], "pdc_dip_snr": rng.random(3 * n)})
    tic = np.arange(3 * n)
    tr = np.arange(3 * n) % 4 != 0
    m = KnownClassModel(feature_cols=list(X.columns), params={"max_iter": 50})
    m.fit(X[tr], y[tr], X[~tr], y[~tr])
    m.fit_typicality(X[tr], y[tr], tic[tr])
    assert set(m.typ) == {"EB", "ROT"} and "pdc_dip_snr" not in m.typ_cols
    q = pd.DataFrame({"pg_a": [3.0, 9.0], "pg_b": [0.0, 0.0], "pdc_dip_snr": [0.5, 0.5]})
    p = m.predict_frame(q)
    assert (p.kc_pred == "EB").all()  # the tree extrapolates: far point is still "EB" ...
    assert p.kc_t_EB[0] > 0.3 and p.kc_t_EB[1] < 0.02  # ... but typicality says it is not one
    assert p.kc_explained[1] < 0.05
