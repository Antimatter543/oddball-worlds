"""Offline tests of the FITS -> shard -> reader path (no network)."""

import io

import numpy as np
import pytest
from astropy.io import fits

from oddball.tess import ingest, store
from oddball.tess.lightcurve import LightCurveError, convert_fits_bytes

SECTOR = 32


def fake_fits(tic: int, n: int = 600, depth: float = 0.0, all_nan: bool = False) -> bytes:
    t = 2174.0 + np.arange(n) * 600 / 86400
    q = np.zeros(n, np.int32)
    q[100] = 32  # momentum dump
    pdc = np.full(n, 5000.0) + np.random.default_rng(tic).normal(0, 1, n)
    if depth:
        pdc[300:330] *= 1 - depth
    pdc[100] = np.nan  # SPOC NaNs flagged cadences
    if all_nan:
        pdc[:] = np.nan
    cols = [
        fits.Column("TIME", "D", array=t),
        fits.Column("PDCSAP_FLUX", "E", array=pdc),
        fits.Column("PDCSAP_FLUX_ERR", "E", array=np.full(n, 1.0)),
        fits.Column("SAP_FLUX", "E", array=pdc * 1.1),
        fits.Column("SAP_BKG", "E", array=np.full(n, 50.0)),
        fits.Column("QUALITY", "J", array=q),
    ]
    h0 = fits.PrimaryHDU()
    h0.header.update(TICID=tic, SECTOR=SECTOR, CAMERA=1, CCD=2, TESSMAG=10.0, RA_OBJ=1.0,
                     DEC_OBJ=2.0, DATA_REL=48, PROCVER="test")  # fmt: skip
    h1 = fits.BinTableHDU.from_columns(cols)
    h1.header["TIMEDEL"] = 600 / 86400
    buf = io.BytesIO()
    fits.HDUList([h0, h1]).writeto(buf)
    return buf.getvalue()


def test_convert_fits_bytes():
    meta, b = convert_fits_bytes(fake_fits(123, depth=0.01))
    assert meta["tic"] == 123 and meta["cadence_s"] == pytest.approx(600)
    assert meta["n_good"] == 599
    assert np.nanmedian(b.flux) == pytest.approx(1.0, abs=1e-3)
    assert np.nanmin(b.flux) == pytest.approx(0.99, abs=2e-3)  # dip survives
    assert 32 in b.extras["qflags"]
    assert np.nanmedian(b.extras["sap_flux"]) == pytest.approx(1.0, abs=1e-3)
    assert np.nanmedian(b.extras["bkg"]) == pytest.approx(50 / 5500, rel=1e-3)


def test_convert_all_nan_is_permanent_failure():
    with pytest.raises(LightCurveError):
        convert_fits_bytes(fake_fits(5, all_nan=True))


def test_parse_script_dedupes_and_checks_sector():
    line = ("curl -f --output 'x' 'https://mast.stsci.edu/api/v0.1/Download/file/?uri=mast:HLSP/"
            "tess-spoc/s{s:04d}/target/0000/0000/0000/{t:04d}/"
            "hlsp_tess-spoc_tess_phot_{t:016d}-s{s:04d}_tess_v1_lc.fits'\n")  # fmt: skip
    text = line.format(s=32, t=7) + line.format(s=32, t=7) + line.format(s=33, t=8)
    tg = ingest.parse_script(text, 32)
    assert [t.tic for t in tg] == [7]


def test_shards_resume_and_reader(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDBALL_DATA", str(tmp_path))
    sdir = tmp_path / "tess-spoc" / f"s{SECTOR:04d}"
    w = ingest.ShardWriter(sdir, SECTOR, shard_size=2)
    for tic in (1, 2, 3):
        w.add(*convert_fits_bytes(fake_fits(tic, depth=0.005 * tic)))
    w.flush()
    assert len(list((sdir / "shards").glob("lc_*.parquet"))) == 2
    log = ingest.FailureLog(sdir / "failures.csv")
    log.log(9, "permanent", "HTTP 404")
    log.log(10, "transient", "timeout")
    log.close()
    done = ingest.done_tics(sdir)
    assert done == {1, 2, 3, 9}  # transient failures are retried

    targets = [ingest.Target(t, f"u{t}") for t in (1, 2, 3, 9, 10, 11, 12)]
    todo = ingest.select_targets(targets, done, limit=5, include_tics={12})
    assert [t.tic for t in todo] == [10, 12]

    # a resumed writer continues the shard numbering instead of overwriting
    assert ingest.ShardWriter(sdir, SECTOR, 2).index == 2

    prov = ingest.finalize(SECTOR, "script-url", 7)
    assert prov["n_converted"] == 3 and prov["sector"] == SECTOR

    lcs = store.load_lightcurves(SECTOR, [3, 1])
    assert set(lcs) == {1, 3}
    lc = lcs[3]
    assert np.nanmin(lc["flux"]) == pytest.approx(0.985, abs=2e-3)
    assert lc["time"][0] == pytest.approx(2174.0, abs=30 / 1440)
    assert len(list(store.iter_lightcurves(SECTOR))) == 3


def test_bin_config_mismatch_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDBALL_DATA", str(tmp_path))
    sdir = tmp_path / "tess-spoc" / f"s{SECTOR:04d}"
    w = ingest.ShardWriter(sdir, SECTOR, shard_size=10)
    w.add(*convert_fits_bytes(fake_fits(1)))
    w.flush()
    ingest.finalize(SECTOR, "u", 1, run={"started_utc": "2026-11-09T00:00:00Z",
                                         "bin_minutes": 30.0, "quality_bitmask": 17087})
    with pytest.raises(SystemExit):
        ingest._check_bin_config(sdir, 20.0, 17087)
    ingest._check_bin_config(sdir, 30.0, 17087)
