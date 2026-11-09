import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from oddball.common.splits import known_positive_tics, split_of
from oddball.ml import data as D


def _row_table(n_rows=6, n_bins=200, bin0=1000, inj=False, tics=None, rng=None):
    rng = rng or np.random.default_rng(0)
    tics = tics if tics is not None else list(range(1000, 1000 + n_rows))
    flux, err, sap = [], [], []
    for i in range(n_rows):
        f = 1 + 1e-3 * rng.standard_normal(n_bins)
        f[50:60] = np.nan  # a gap
        flux.append(f.astype(np.float32))
        err.append(np.full(n_bins, 1e-3, np.float32))
        sap.append((f * 1.01).astype(np.float32))
    cols = {
        "tic": pa.array(tics, pa.int64()),
        "sector": pa.array([32] * n_rows, pa.int16()),
        "bin0": pa.array([bin0 + i for i in range(n_rows)], pa.int64()),
        "flux": pa.array(flux, pa.list_(pa.float32())),
        "flux_err": pa.array(err, pa.list_(pa.float32())),
        "sap_flux": pa.array(sap, pa.list_(pa.float32())),
    }
    if inj:
        cols["inj_id"] = pa.array([f"inj{i}" for i in range(n_rows)])
        cols["host_tic"] = pa.array(tics, pa.int64())
    return pa.table(cols)


def test_rows_to_grid_places_and_keeps_gaps():
    t = _row_table(n_rows=3, n_bins=200, bin0=1000)
    g = D.rows_to_grid(t.to_batches()[0], origin=1000, length=256)
    assert g["flux"].shape == (3, 256)
    for i in range(3):
        assert np.isnan(g["flux"][i, :i]).all()  # leading offset = bin0 - origin
        assert np.isfinite(g["flux"][i, i])
        assert np.isnan(g["flux"][i, i + 50 : i + 60]).all()  # gap stays a gap (no interpolation)
        assert np.isnan(g["flux"][i, i + 200 :]).all()
    # truncation past the grid end does not raise
    g2 = D.rows_to_grid(t.to_batches()[0], origin=900, length=256)
    assert np.isfinite(g2["flux"][0, 100:150]).all() and np.isfinite(g2["flux"][0, 160:256]).all()


def test_standardise_robust_scale_and_mask():
    rng = np.random.default_rng(1)
    f = 1 + 2e-3 * rng.standard_normal((4, 500))
    f[:, 100:120] = np.nan
    f[0, 300] = 0.7  # a 30% single-bin dip: compressed, not clipped, stays most negative
    st = D.standardise(f, np.full_like(f, 2e-3), f.copy())
    assert np.isnan(st["pdc"][:, 100:120]).all() and np.isnan(st["err"][:, 100:120]).all()
    assert abs(np.nanstd(st["pdc"][1]) - 1.0) < 0.1  # robust sigma units
    assert np.nanargmin(st["pdc"][0]) == 300 and -40 < st["pdc"][0, 300] < -30  # 10*asinh(-15)
    assert np.allclose(st["log_scatter"], np.log10(2e-3), atol=0.1)  # MAD noise on 480 pts
    assert (st["n_obs"] == 480).all()


def test_make_inputs_shapes_and_masking():
    B, L = 3, D.GRID_LEN
    x = torch.randn(B, 3, L)
    x[:, 2] = 1.0
    x[0, :, 10:40] = float("nan")
    inp, tgt, _err, obs, _sobs = D.make_inputs(x)
    assert inp.shape == (B, L // D.PATCH, D.PATCH * 4)
    assert tgt.shape == (B, 2, L) and obs.shape == (B, L)
    assert not obs[0, 10:40].any() and obs[1].all()
    assert (tgt[0, :, 10:40] == 0).all()  # unobserved bins zeroed, not interpolated
    feats = inp.reshape(B, L, 4)
    assert (feats[0, 10:40, 3] == 0).all() and (feats[1, :, 3] == 1).all()
    # an external (common-gap) mask removes bins for everyone
    om = torch.ones(L, dtype=torch.bool)
    om[100:110] = False
    _, _, _, obs2, _ = D.make_inputs(x, om)
    assert not obs2[:, 100:110].any()


def test_augment_never_time_reverses():
    B, L = 8, D.GRID_LEN
    ramp = torch.arange(L, dtype=torch.float32)[None, :].repeat(B, 1)
    x = torch.stack([ramp, ramp, torch.full_like(ramp, 1e-6)], 1)
    g = torch.Generator().manual_seed(0)
    y = D.augment(x, g, noise_max=0.0, amp_jitter=0.0)
    assert y.shape == x.shape
    for b in range(B):
        v = y[b, 0][torch.isfinite(y[b, 0])]
        assert len(v) > L // 2
        assert (torch.diff(v) > 0).all()  # order preserved: shifted/cropped, never reversed


def test_block_masks():
    rng = np.random.default_rng(0)
    m = D.contiguous_block_mask(16, 80, 0.4, rng)
    assert ((m.sum(1) >= 32) & (m.sum(1) <= 32 + 8)).all()
    sm = D.scoring_masks(80, 5, rng)
    assert sm.shape == (5, 80)
    assert (sm.sum(0) == 2).all()  # every patch hidden exactly twice
    assert (sm.sum(1) == 32).all()  # each mask hides 40%


def test_cache_build_and_no_leakage(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDBALL_DATA", str(tmp_path / "data"))
    known = sorted(known_positive_tics())[:2]
    tics = list(range(5000, 5200)) + known
    src = tmp_path / "data" / "injections" / "vt" / "s0032" / "val"
    src.mkdir(parents=True)
    pq.write_table(_row_table(len(tics), 300, 104363, inj=True, tics=tics), src / "lc_00000.parquet")
    cdir = D.build_cache(src, D.cache_dir_for(src), log=lambda s: None)
    c = D.Cache(cdir)
    assert c.x.shape == (len(tics), 3, D.GRID_LEN) and c.x.dtype == np.float16
    assert c.index.inj_id.notna().all()
    assert list(c.index.split) == [split_of(t) for t in tics]
    # injected rows are never training rows; known/test never either
    assert len(D.training_rows(c, "train")) == 0
    # real rows: only split == train, never known/test/val
    c.index["inj_id"] = None
    tr = D.training_rows(c, "train")
    assert set(c.index.loc[tr, "split"]) == {"train"}
    assert not set(known) & set(c.index.loc[tr, "tic"])
    with pytest.raises(AssertionError):
        D.training_rows(c, "test")
    # idempotent rebuild
    assert D.build_cache(src, cdir, log=lambda s: None) == cdir


def test_multicache_global_rows_and_batches(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDBALL_DATA", str(tmp_path / "data"))
    dirs = []
    for k, (n, sec) in enumerate(((30, 33), (20, 38))):
        src = tmp_path / "data" / "tess-spoc" / f"s{sec:04d}" / "shards"
        src.mkdir(parents=True)
        t = _row_table(n, 300, 1000, tics=list(range(7000 + 100 * k, 7000 + 100 * k + n)))
        t = t.set_column(1, "sector", pa.array([sec] * n, pa.int16()))
        pq.write_table(t, src / "lc_00000.parquet")
        dirs.append(D.build_cache(src.parent, D.cache_dir_for(src.parent), origin=1000,
                                  log=lambda s: None))  # fmt: skip
    assert [d.name for d in dirs] == ["tess-spoc__s0033", "tess-spoc__s0038"]
    mc = D.MultiCache(dirs)
    assert len(mc) == 50 and mc.sectors == [33, 38]
    cid, loc = mc.split_global(np.array([0, 29, 30, 49]))
    assert list(cid) == [0, 0, 1, 1] and list(loc) == [0, 29, 0, 19]
    seen = []
    for g, c, x, ctx in mc.iter_batches(np.arange(50), 16, shuffle=True, seed=1):
        cc, ll = mc.split_global(g)
        assert (cc == c).all() and x.shape == (len(g), 3, D.GRID_LEN) and ctx.shape == (len(g), 3)
        for i in range(len(g)):  # the batch row really is that cache's row
            ref = np.asarray(mc.caches[cc[i]].x[ll[i]], dtype=np.float32)
            assert np.array_equal(np.isnan(ref), np.isnan(x[i]))
            assert np.allclose(np.nan_to_num(ref), np.nan_to_num(x[i]))
        seen += list(g)
    assert sorted(seen) == list(range(50))
    assert set(mc.index_of(mc.training_rows("train")).split) <= {"train"}
    masks = D.sector_gap_masks(mc, np.arange(50))
    assert masks.shape == (2, D.GRID_LEN)
    assert masks[:, 100:200].all() and not masks[:, 0].any()  # data from bin ~0..300 only


def test_signal_dominated():
    x = np.full((2, 3, D.GRID_LEN), np.nan, np.float32)
    x[:, 0, :600] = 0.0
    x[0, 2, :600] = 0.1  # errors far below the robust scatter: variability dominates
    x[1, 2, :600] = 1.0  # white-noise dominated
    assert list(D.signal_dominated(x)) == [True, False]
