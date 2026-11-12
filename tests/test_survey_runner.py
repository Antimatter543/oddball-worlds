"""Track C survey runner: shard split, target order and the S3 key mapping (no network)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SURVEY = Path(__file__).resolve().parents[1] / "spikes" / "mdwarf_stack" / "survey"
sys.path.insert(0, str(SURVEY))

import download  # noqa: E402
import run_survey as R  # noqa: E402


def _targets(tmp_path, n=103):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"tic": rng.permutation(10_000)[:n] + 1,
                       "n_sectors": rng.integers(2, 46, n), "mass": 0.3, "rad": 0.3})
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    return p


def test_order_and_min_sectors(tmp_path):
    t = R.load_targets(_targets(tmp_path), 6)
    assert (t["n_sectors"] >= 6).all()
    key = list(zip(-t["n_sectors"], t["_tic"]))
    assert key == sorted(key)  # most sectors first, ties by TIC


def test_shards_partition_the_list(tmp_path):
    t = R.load_targets(_targets(tmp_path), 6)
    for of in (1, 2, 5, 7):
        parts = [R.shard_of(t, i, of) for i in range(of)]
        tics = np.concatenate([p["_tic"].to_numpy() for p in parts])
        assert sorted(tics) == sorted(t["_tic"])  # no gap, no duplicate
        assert max(map(len, parts)) - min(map(len, parts)) <= 1
        assert R.shard_of(t, 0, of)["_tic"].tolist() == parts[0]["_tic"].tolist()  # stable


def test_shard_dir():
    b = Path("/x")
    assert R.shard_dir(b, 0, 1) == b
    assert R.shard_dir(b, 3, 12) == b / "shards" / "shard003of012"


def test_s3_keys():
    u = "mast:TESS/product/tess2018234235059-s0002-0000000012421862-0121-s_lc.fits"
    assert download.s3_key(u) == ("tess/public/tid/s0002/0000/0000/1242/1862/"
                                  "tess2018234235059-s0002-0000000012421862-0121-s_lc.fits")
    h = ("mast:HLSP/tess-spoc/s0029/target/0000/0000/1242/1862/"
         "hlsp_tess-spoc_tess_phot_0000000012421862-s0029_tess_v1_lc.fits")
    assert download.s3_key(h) == "mast/hlsp/" + h[len("mast:HLSP/"):]
    assert download.s3_key("mast:HLSP/qlp/x.fits") is None
