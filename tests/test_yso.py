import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from oddball.common.splits import known_positive_tics, split_of
from oddball.inject import apply_model, draw_params
from oddball.inject.sampling import DEPTH_RANGE
from oddball.inject.yso import (
    NULL_KIND,
    YSO_KINDS,
    build_host_set,
    null_params,
    plan_rows,
    vsx_young,
    young_host_pool,
)
from oddball.tess.ingest import LC_SCHEMA

BIN = 30.0 / 1440.0


def _tics_by_split(n_each=6, start=10_000_000):
    out = {"train": [], "val": [], "test": []}
    t = start
    while min(len(v) for v in out.values()) < n_each:
        s = split_of(t)
        if s in out and len(out[s]) < n_each:
            out[s].append(t)
        t += 1
    return out


def test_vsx_young_types():
    for t in ("YSO", "TTS/ROT", "WTTS", "INB:", "UXOR+DSCT", "EA+TTS", "DIP", "IN"):
        assert vsx_young(t), t
    for t in ("ROT", "EA", "RCB", "BOY", "DSCT", "", None):
        assert not vsx_young(t), t


def _pool_fixture():
    sp = _tics_by_split()
    known = sorted(known_positive_tics())[:1]
    rows = []
    for k, t in enumerate(t for ts in sp.values() for t in ts):  # 2 deg apart
        rows.append({"tic": t, "ra": 10.0 + 2 * k, "dec": -1.0, "tmag": 12.0, "camera": 1,
                     "ccd": 2})  # fmt: skip
    rows.append({"tic": known[0], "ra": 85.0, "dec": -1.0, "tmag": 12.0, "camera": 1, "ccd": 2})
    meta = pd.DataFrame(rows)
    vsx = pd.DataFrame({"tic": [t for ts in sp.values() for t in ts] + [known[0]],
                        "Name": "v", "Type": "YSO", "sep_arcsec": 1.0})  # fmt: skip
    return sp, known, meta, vsx


def test_young_host_pool_split_pure_and_exclusions():
    sp, known, meta, vsx = _pool_fixture()
    # one val star is not young (ROT), one only a 30" VSX neighbour (outside 21"); one test star
    # is young only in Gaia; one val star sits 60" from the known positive; one is an old host
    v = vsx.copy()
    v.loc[v.tic == sp["val"][0], "Type"] = "ROT"
    v.loc[v.tic == sp["val"][1], "sep_arcsec"] = 30.0
    v = v[v.tic != sp["test"][0]]
    gaia = pd.DataFrame({"tic": [sp["test"][0], sp["test"][1]], "Class": ["YSO", "RS"],
                         "Source": [1, 2], "sep_arcsec": [0.5, 0.5]})  # fmt: skip
    meta.loc[meta.tic == sp["val"][2], ["ra", "dec"]] = (85.0 + 60 / 3600 / np.cos(np.radians(1)),
                                                        -1.0)  # fmt: skip
    pool, counts = young_host_pool(meta, v, gaia, exclude_tics=[sp["val"][3]])
    assert set(pool.split) <= {"val", "test"}
    assert not pool.tic.isin(known).any() and not pool.tic.isin(sp["train"]).any()
    assert set(pool.tic) == set(sp["val"][4:]) | set(sp["test"])
    src = pool.set_index("tic").yso_source
    assert src[sp["test"][0]] == "gaia" and src[sp["test"][1]] == "vsx"
    assert counts["near_known"] == 1 and counts["earlier_hosts"] == 1
    assert (pool.sep_known > 120).all()


def test_plan_rows_every_host_every_kind_once_per_replica():
    hosts = pd.DataFrame({"tic": [1, 2, 3]})
    plan = plan_rows(hosts, replicas=10)
    rep = plan[plan.replica >= 0]
    assert not rep.duplicated(["tic", "replica"]).any()
    assert (rep.groupby("tic").kind.nunique() == len(YSO_KINDS)).all()
    nul = plan[plan.kind == NULL_KIND]
    assert sorted(nul.tic) == [1, 2, 3] and (nul.replica == -1).all()


@pytest.mark.parametrize("kind", ["aperiodic_dipper", "exocomet", "dusty_tail"])
def test_depth_range_override_and_default_unchanged(kind):
    t = (104363 + np.arange(1200) + 0.5) * BIN
    valid = np.ones_like(t, bool)
    for seed in range(10):
        p = draw_params(kind, np.random.default_rng(seed), t, valid, depth_range=(0.01, 0.02))
        assert 0.01 <= p["depth"] <= 0.02
        a = draw_params(kind, np.random.default_rng(seed), t, valid)
        b = draw_params(kind, np.random.default_rng(seed), t, valid, depth_range=None)
        assert a == b
    if kind != "dusty_tail":
        assert DEPTH_RANGE[0] <= a["depth"] <= DEPTH_RANGE[1]


def test_null_params_leave_the_light_curve_unchanged():
    t = (104363 + np.arange(500) + 0.5) * BIN
    f = (1 + 1e-3 * np.random.default_rng(0).standard_normal(500)).astype(np.float32)
    fi, si, m = apply_model(f, f, t, null_params(t), BIN, crowdsap=0.7)
    assert np.array_equal(fi, f) and np.array_equal(si, f) and (m == 1).all()


def _fake_sector(root, tics):
    d = root / "tess-spoc" / "s0032"
    (d / "shards").mkdir(parents=True)
    rng = np.random.default_rng(3)
    n = 400
    rows, meta = [], []
    for k, t in enumerate(tics):
        f = 1 + 2e-3 * rng.standard_normal(n)
        f[100:110] = np.nan
        rows.append({"tic": t, "sector": 32, "bin0": 104363, "flux": f.astype(np.float32),
                     "flux_err": np.full(n, 2e-3, np.float32), "npts": np.full(n, 3, np.uint16),
                     "qflags": np.zeros(n, np.int32), "sap_flux": f.astype(np.float32),
                     "bkg": np.zeros(n, np.float32)})  # fmt: skip
        meta.append({"tic": t, "sector": 32, "camera": 1 + k % 2, "ccd": 2, "tmag": 11.0 + k % 3,
                     "ra": 80.0, "dec": -1.0, "radius": 1.0, "crowdsap": 0.9, "bin0": 104363,
                     "n_bins": n, "bin_minutes": 30.0, "shard": 0})  # fmt: skip
    pq.write_table(pa.Table.from_pylist(rows, schema=LC_SCHEMA), d / "shards" / "lc_00000.parquet")
    pq.write_table(pa.Table.from_pylist(meta), d / "shards" / "meta_00000.parquet")
    pq.write_table(pa.Table.from_pylist(meta), d / "metadata.parquet")
    return d


def test_build_host_set_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDBALL_DATA", str(tmp_path / "data"))
    sp = _tics_by_split(4)
    sd = _fake_sector(tmp_path / "data", sp["val"] + sp["test"])
    hosts = pd.DataFrame({"tic": sp["val"], "split": "val", "yso_type": "YSO"})
    prov = build_host_set(32, "val", hosts, replicas=5, log=lambda s: None)
    out = tmp_path / "data" / "injections" / "yso1" / "s0032" / "val"
    inj = pd.read_parquet(out / "injections.parquet")
    assert len(inj) == 4 * 6 and prov["n_null"] == 4 and inj.inj_id.is_unique
    assert set(inj.host_tic) == set(sp["val"]) and (inj.split == "val").all()
    astro = inj[~inj.is_null]
    assert astro.depth.between(0.00173, 0.173).all()
    assert (astro.groupby("host_tic").kind.nunique() == len(YSO_KINDS)).all()
    assert (inj[inj.is_null].kind == NULL_KIND).all() and (inj[inj.is_null].replica == -1).all()
    lc = pq.read_table(out / "shards" / "lc_00000.parquet").to_pandas()
    real = pq.read_table(sd / "shards" / "lc_00000.parquet").to_pandas().set_index("tic")
    for r in lc.merge(inj[["inj_id", "is_null", "depth"]], on="inj_id").itertuples():
        f0 = np.asarray(real.loc[r.tic, "flux"])
        f = np.asarray(r.flux)
        assert np.array_equal(np.isnan(f), np.isnan(f0))  # gaps stay gaps
        if r.is_null:
            assert np.array_equal(f, f0, equal_nan=True)
        else:
            assert np.nanmin(f / f0) < 1 - 0.5 * r.depth
    assert set(pd.read_parquet(out / "hosts.parquet").tic) == set(sp["val"])
    assert json.loads((out / "provenance.json").read_text())["replicas"] == 5
    with pytest.raises(ValueError):
        build_host_set(32, "test", hosts, replicas=1, log=lambda s: None)
