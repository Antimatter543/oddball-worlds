import numpy as np
import pandas as pd

from oddball.score import veto as V
from oddball.score import veto3 as V3

L = 1245


def _b(n=40, seed=0):
    rng = np.random.default_rng(seed)
    f = 1 + 1e-3 * rng.standard_normal((n, L))
    f[:, 560:640] = np.nan  # mid-sector gap
    sap = f * 1.0 + 1e-3 * rng.standard_normal((n, L))
    bkg = 0.2 + 1e-4 * rng.standard_normal((n, L))
    bkg[np.isnan(f)] = np.nan
    return {"flux": f, "sap_flux": sap, "bkg": bkg, "qflags": np.zeros((n, L), np.int32)}


def _masks(dump_bins=()):
    return V.SysMasks(np.zeros((len(V.SERIES), 16, L), bool),
                      np.zeros((len(V.SERIES), 16, V.NCELL**2, L), bool),
                      np.zeros((16, L), bool), np.asarray(dump_bins, np.int64))  # fmt: skip


def _labels(b, dump_bins=(), crowd=None):
    n = len(b["flux"])
    ci = np.zeros(n, np.int64)
    sig = V3.batch_signal(b)
    parts, _ = V.events_stage1(sig.snr, ci, ci, _masks(dump_bins))
    ev = pd.concat(parts, ignore_index=True)
    me = V3.event_measures(ev, sig, np.ones(n) if crowd is None else crowd)
    lab = V3.measure_labels(ev, me, np.full(16, np.inf), ci)
    lab["dumpedge"] = V3.dumpedge_label(ev, np.asarray(dump_bins, np.int64))
    return ev, me, lab


def test_background_explains_dip_but_not_real_dip():
    b = _b()
    # star 0: an object in the background pixels; bkg rises, SAP and PDCSAP dip, raw flat
    b["bkg"][0, 300:304] += 0.01
    b["sap_flux"][0, 300:304] -= 0.01
    b["flux"][0, 300:304] -= 0.01
    # star 1: a real dip (the star fades; bkg unchanged)
    b["sap_flux"][1, 300:304] -= 0.01
    b["flux"][1, 300:304] -= 0.01
    ev, _, lab = _labels(b)
    strong = (ev.w == 0) & (ev.sign == 0) & (ev.peak > 8)
    e0 = strong & (ev.row == 0)
    e1 = strong & (ev.row == 1)
    assert e0.any() and e1.any()
    assert lab["bkg"][e0.to_numpy()].all()
    assert not lab["bkg"][e1.to_numpy()].any()


def test_pdc_only_event_is_labelled_sap_event_with_crowding_is_not():
    b = _b(seed=1)
    b["flux"][0, 300:308] -= 0.01  # PDCSAP only
    b["flux"][1, 300:308] -= 0.01  # PDCSAP and SAP (CROWDSAP 0.5: half the SAP depth)
    b["sap_flux"][1, 300:308] -= 0.005
    crowd = np.ones(40)
    crowd[1] = 0.5
    ev, _, lab = _labels(b, crowd=crowd)
    pdc = (ev.ch == 0) & (ev.sign == 0) & (ev.peak > 8)
    assert lab["pdc"][(pdc & (ev.row == 0)).to_numpy()].all()
    assert not lab["pdc"][(pdc & (ev.row == 1)).to_numpy()].any()
    # a SAP-only dip (PDC suppressed it, as for PDS 110) is never a pdc artefact
    assert not lab["pdc"][(ev.ch == 1).to_numpy()].any()


def test_step_at_dump_is_dumpedge():
    b = _b(seed=2)
    step = 0.006 * np.exp(-np.arange(L - 400) / 100.0)  # step at the dump, ~2-d recovery
    b["flux"][3, 400:] -= step
    b["sap_flux"][3, 400:] -= step
    ev, _, lab = _labels(b, dump_bins=[399])
    e = (ev.row == 3) & (ev.sign == 0) & (ev.peak > 8)
    assert e.any() and lab["dumpedge"][e.to_numpy()].any()
    ev2, _, lab2 = _labels(b, dump_bins=[900])
    e2 = (ev2.row == 3) & (ev2.sign == 0) & (ev2.peak > 8)
    assert e2.any() and not lab2["dumpedge"][e2.to_numpy()].any()


def test_edge_counts_mark_gap_and_ends():
    f = np.ones((1, 50))
    f[0, :3] = np.nan  # padding before the light curve
    f[0, 20:40] = np.nan  # a long gap
    f[0, 10] = np.nan  # a one-bin hole is not an edge
    e = V3.edge_counts(f, gap_bins=15)
    edges = np.flatnonzero(np.diff(e[0]))
    assert edges.tolist() == [3, 19, 40, 49]


def test_find_tracks_line_vs_scatter():
    # four stars hit one after another by an object moving 1 px per bin
    t = np.array([100, 104, 109, 115])
    x = 500.0 + 1.0 * (t - 100) + np.array([0.5, -1.0, 2.0, -0.5])
    y = 800.0 + 0.3 * (t - 100)
    tr = V3.find_tracks(x, y, t, np.array([1, 2, 3, 4]))
    assert len(tr) and (tr.n_star >= 3).all()
    # the same times at scattered positions: no track
    rng = np.random.default_rng(0)
    tr2 = V3.find_tracks(rng.uniform(0, 2000, 4), rng.uniform(0, 2000, 4), t,
                         np.array([1, 2, 3, 4]))  # fmt: skip
    assert len(tr2) == 0
    # one star seen three times is not a track
    tr3 = V3.find_tracks(x[:3], y[:3], t[:3], np.array([1, 1, 1]))
    assert len(tr3) == 0


def test_track_windows_cover_passing_star_only():
    tr = pd.DataFrame({"x0": [500.0], "y0": [800.0], "t0": [100], "vx": [1.0], "vy": [0.0],
                       "t_lo": [100], "t_hi": [120], "n_star": [3]})  # fmt: skip
    sx = np.array([510.0, 510.0])
    sy = np.array([803.0, 830.0])  # 3 px and 30 px from the line
    w = V3.track_windows(tr, sx, sy, np.array([7, 8]))
    assert w.lc.tolist() == [7]
    assert w.lo.iloc[0] <= 110 <= w.hi.iloc[0]


def test_column_event_labelled_spread_event_not():
    rng = np.random.default_rng(3)
    n = 3000
    stars = pd.DataFrame({"ccd": 0, "col": rng.uniform(45, 2092, n)})
    stars.loc[:5, "col"] = 1000.0 + np.arange(6) * 0.3  # six stars on one column
    base = {"ccd": 0, "ch": 0, "sign": 0, "w": 0}
    col_ev = pd.DataFrame({"lc": np.arange(6), "col": stars.col[:6].to_numpy(), "start": 300,
                           "end": 304, "peak_col": 302, **base})  # fmt: skip
    spread = pd.DataFrame({"lc": np.arange(100, 106), "col": [200.0, 600, 900, 1300, 1500, 1900],
                           "start": 300, "end": 304, "peak_col": 302, **base})  # fmt: skip
    noise = pd.DataFrame({"lc": np.arange(1000, 1300), "col": rng.uniform(45, 2092, 300),
                          "start": rng.integers(0, 1200, 300), **base})  # fmt: skip
    noise["end"] = noise.start + 3
    noise["peak_col"] = noise.start + 1
    ref = pd.concat([col_ev, spread, noise], ignore_index=True)
    lab = V3.column_labels(ref, ref, stars, L)
    assert lab[:6].all()
    assert not lab[6:12].any()
    assert lab[12:].mean() < 0.01


def test_cmv2_configuration_matches_cmv2():
    b = _b(seed=5)
    b["flux"][:30, 300:306] -= 0.01
    b["flux"][0, 800:806] -= 0.006
    masks = _masks()
    masks.cm_ccd[:, 0, 299:307] = True
    ci = np.zeros(40, np.int64)
    ref = V.summarise_events(V.events_batch(V.batch_snr(b), ci, ci, masks), 40, np.arange(L))
    sig = V3.batch_signal(b)
    parts, members = V.events_stage1(sig.snr, ci, ci, masks)
    got = V3.summarise(V.events_stage2(parts, members, 40), 40, np.arange(L), "cmv2")
    for k in ("res_snr", "sys_snr", "res_snr_pdc", "sys_snr_pdc"):
        assert np.allclose(ref[f"cmv2_{k}"], got[f"cmv2_{k}"])


def test_null2_asteroid_response_annulus_vs_aperture():
    from oddball.inject import nulls2 as N

    t = np.arange(0, 2, 1 / 48) + 1 / 96
    v = 40.0 / 21.0 * 24.0  # 40 arcsec/h in px/day
    # passes 3.5 px from the star (background pixels): background only, no aperture flux
    ap, bk = N._asteroid_response(t, 1 / 48, 0.0, 3.5, -v * 1.0, 0.0, v, 0.0, 0.0)
    assert bk.max() > 0.5 and ap.max() < 0.05
    # crosses the star: aperture flux, flanked by background on both sides (dip-spike-dip)
    ap2, bk2 = N._asteroid_response(t, 1 / 48, 0.0, 0.0, -v * 1.0, 0.0, v, 0.0, 0.0)
    k = int(np.argmax(ap2))
    assert ap2[k] > 0.3 and bk2[k] < 0.2
    assert bk2[: k - 1].max() > 0.5 and bk2[k + 2 :].max() > 0.5


def test_null2_step_model_starts_at_dump():
    from oddball.inject import nulls2 as N

    t = np.arange(0, 3, 1 / 48) + 1 / 96
    m = N._step_model(t, 1 / 48, 1.0, -0.01, 1.0)
    assert np.allclose(m[t < 0.98], 1.0)
    assert m[(t > 1.02) & (t < 1.1)].max() < 0.992


def test_flank_labels_crossing_spike_not_flare():
    b = _b(seed=4)
    # star 0: dip-spike-dip, a body crossing the background pixels, the aperture, then the
    # background pixels again (raw aperture flux moves only during the spike)
    for lo in (290, 310):
        b["bkg"][0, lo : lo + 4] += 0.01
        b["sap_flux"][0, lo : lo + 4] -= 0.01
        b["flux"][0, lo : lo + 4] -= 0.01
    b["sap_flux"][0, 300:303] += 0.05
    b["flux"][0, 300:303] += 0.05
    # star 1: a flare that leaks into the background pixels (bkg rises with the flare)
    b["sap_flux"][1, 300:303] += 0.05
    b["flux"][1, 300:303] += 0.05
    b["bkg"][1, 300:303] += 0.005
    ev, _, lab = _labels(b)
    parts = [g for _, g in ev.groupby(["ch", "sign", "w"], sort=False)]
    labs = [{k: v[g.index.to_numpy()] for k, v in lab.items()} for g in parts]
    fl = np.zeros(len(ev), bool)
    for g, f in zip(parts, V3.flank_labels(parts, labs), strict=True):
        fl[g.index.to_numpy()] = f
    spike = (ev.sign == 1) & (ev.w == 0) & (ev.peak > 8)
    assert (spike & (ev.row == 0)).any() and (spike & (ev.row == 1)).any()
    assert fl[(spike & (ev.row == 0)).to_numpy()].all()
    assert not fl[(spike & (ev.row == 1)).to_numpy()].any()
    assert not fl[(ev.sign == 0).to_numpy()].any()  # only bumps


def test_half_max_width_counts_core_not_wings():
    snr = np.zeros((2, 50))
    snr[0, 10:30] = 4.0  # long low run
    snr[0, 18:21] = 100.0  # 3-bin core
    snr[1, 10:30] = 20.0  # flat 20-bin run
    ev = pd.DataFrame({"row": [0, 1], "start": [10, 10], "end": [29, 29],
                       "peak": [100.0, 20.0]})  # fmt: skip
    assert V3.half_max_width(snr, ev).tolist() == [3, 20]
