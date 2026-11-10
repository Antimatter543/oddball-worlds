import numpy as np
import pandas as pd

from oddball.score import veto as V

BIN = 30.0 / 1440.0
L = 1245


def _batch(n=40, seed=0):
    rng = np.random.default_rng(seed)
    f = 1 + 1e-3 * rng.standard_normal((n, L))
    f[:, 560:640] = np.nan  # mid-sector gap
    return f


def _masks(cm_bins=(), dump_bins=()):
    cm = np.zeros((len(V.SERIES), 16, L), bool)
    for b in cm_bins:
        cm[:, 0, b] = True
    return V.SysMasks(cm, np.zeros((len(V.SERIES), 16, V.NCELL**2, L), bool),
                      np.zeros((16, L), bool), np.asarray(dump_bins, np.int64))


def _snr(f):
    return V.batch_snr({"flux": f, "sap_flux": f.copy()})


def test_segments_split_at_row_boundaries():
    a = np.zeros((2, 5), bool)
    a[0, 3:] = True
    a[1, :2] = True
    _, seg, starts, rows = V._segments(a)
    assert len(starts) == 2 and rows.tolist() == [0, 1]
    assert seg.tolist() == [0, 0, 1, 1]


def test_shared_event_is_systematic_private_dip_survives():
    f = _batch()
    f[:30, 300:306] -= 0.01  # shared by 30 of 40 stars: instrument
    f[0, 800:806] -= 0.006  # star 0 also has its own dip
    masks = _masks(cm_bins=range(299, 307))
    ev = V.events_batch(_snr(f), np.zeros(len(f), np.int64), np.zeros(len(f), np.int64), masks)
    s = V.summarise_events(ev, len(f), np.arange(L) * BIN)
    # the shared dip is systematic on every star that has it
    assert (s.cmv2_sys_snr[:30] > 10).all()
    # star 0 keeps its own dip as the astrophysical residual; star 1 has nothing left
    assert s.cmv2_res_snr[0] > 10
    assert s.cmv2_res_snr[1] < 6
    flag = V.veto_flag(s, t_sys=8.0, t_res=6.0)
    assert not flag[0] and flag[1] and not flag[35]


def test_dump_shaped_event_is_systematic():
    f = _batch(seed=3)
    f[2, 400:408] -= 0.008 * np.exp(-np.arange(8) / 3.0)
    ev = V.events_batch(_snr(f), np.zeros(len(f), np.int64), np.zeros(len(f), np.int64),
                        _masks(dump_bins=[399]))
    e = ev[(ev.row == 2) & (ev.peak > 8)]
    assert len(e) and e.sys.all() and (e.cause == 4).any()
    # the same dip far from any dump is not
    ev2 = V.events_batch(_snr(f), np.zeros(len(f), np.int64), np.zeros(len(f), np.int64),
                         _masks(dump_bins=[1000]))
    e2 = ev2[(ev2.row == 2) & (ev2.peak > 8)]
    assert len(e2) and not e2.sys.any()


def test_threshold_cm_needs_excess_and_count():
    den = np.full((1, 1, 200), 1000.0)
    num = np.full((1, 1, 200), 30.0)
    num[0, 0, 50] = 200.0  # 20% of stars: common-mode
    num[0, 0, 60] = 34.0  # within the noise of 3%
    cm = V.threshold_cm(num, den)
    assert cm[0, 0, 50] and not cm[0, 0, 60] and cm.sum() == 1


def test_group_sum_matches_loop():
    rng = np.random.default_rng(1)
    a = rng.random((50, 7))
    k = rng.integers(0, 5, 50)
    g = V.group_sum(a, k, 6)
    for j in range(6):
        assert np.allclose(g[j], a[k == j].sum(0))


def test_red_noise_rescaling_keeps_white_noise_snr():
    f = _batch(n=20, seed=4)
    s = _snr(f)
    v = s[(0, 0, 0)]
    rs = 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v, 1, keepdims=True)), 1)
    assert np.all((rs > 0.8) & (rs < 1.2))
    df = pd.DataFrame({"cmv2_sys_snr": [9.0, 9.0, 2.0], "cmv2_res_snr": [5.0, 7.0, 3.0]})
    assert V.veto_flag(df, 8.0, 6.0).tolist() == [True, False, False]


def test_short_real_dip_next_to_systematic_window_is_kept():
    # a wide-box view of the dip runs into a common-mode window; the 2-h event must survive
    f = _batch(seed=6)
    f[0, 800:806] -= 0.008
    masks = _masks(cm_bins=range(812, 880))
    ev = V.events_batch(_snr(f), np.zeros(len(f), np.int64), np.zeros(len(f), np.int64), masks)
    e = ev[(ev.row == 0) & (ev.w == 0) & (ev.sign == 0) & (ev.peak > 10)]
    assert len(e) and not e.sys.any()
