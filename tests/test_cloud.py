"""Offline tests of the Kaggle helpers and the sector-dir verifier."""

import ast
import json
from pathlib import Path

import pytest

from oddball.tess import cloud, ingest
from oddball.tess.lightcurve import convert_fits_bytes
from oddball.tess.verify import verify_sector_dir
from tests.test_ingest import SECTOR, fake_fits

SHA = "e98f7679d64c02d9eb32e10d502a21dc4de6545c"
TEMPLATE = Path(__file__).resolve().parents[1] / "scripts" / "kaggle" / "ingest_kernel.py"


# --------------------------------------------------------------------------------------------
# kernel metadata + rendering


def test_kernel_metadata_private_cpu_internet():
    m = cloud.kernel_metadata("someone", 33)
    assert m["id"] == "someone/oddball-ingest-s0033"
    assert m["is_private"] is True and m["enable_internet"] is True
    assert m["enable_gpu"] is False and m["enable_tpu"] is False
    assert m["kernel_type"] == "script" and m["code_file"] == cloud.KERNEL_FILE
    # Kaggle derives the slug from the title; they must agree or pushes go to a surprise slug.
    assert m["title"].lower().replace(" ", "-") == m["id"].split("/")[1]
    assert cloud.kernel_slug(33, 2000) == "oddball-ingest-s0033-l2000"
    with pytest.raises(ValueError):
        cloud.kernel_metadata("bad/user", 33)


def test_kernel_config_validates():
    c = cloud.kernel_config(33, SHA, "u/k", "2026-11-09T00:00:00Z", include_tics=[5, 3])
    assert c["include_tics"] == [3, 5] and c["limit"] is None
    with pytest.raises(ValueError):
        cloud.kernel_config(33, "e98f767", "u/k", "t")  # short SHA is not a pin
    with pytest.raises(ValueError):
        cloud.kernel_config(33, SHA, "u/k", "t", deadline_hours=12.5)


def test_render_real_template():
    cfg = cloud.kernel_config(33, SHA, "u/k", "2026-11-09T00:00:00Z", limit=2000)
    code = cloud.render_kernel(TEMPLATE.read_text(), cfg)
    assert cloud.CONFIG_MARKER not in code
    tree = ast.parse(code)
    assigns = [n for n in tree.body if isinstance(n, ast.Assign)
               and getattr(n.targets[0], "id", None) == "CONFIG"]  # fmt: skip
    assert len(assigns) == 1
    assert ast.literal_eval(assigns[0].value) == json.loads(json.dumps(cfg))
    with pytest.raises(ValueError):
        cloud.render_kernel("print(1)\n", cfg)


def test_output_relpath():
    assert cloud.output_relpath("tess-spoc/s0033/shards/lc_00000.parquet", 33) == (
        "shards/lc_00000.parquet"
    )
    assert cloud.output_relpath("tess-spoc/s0033/provenance.json", 33) == "provenance.json"
    for bad in ("tess-spoc/s0034/provenance.json", "tess-spoc/s0033", "/etc/passwd",
                "tess-spoc/s0033/../../x", "kaggle.log", "__results__.html"):  # fmt: skip
        assert cloud.output_relpath(bad, 33) is None, bad


# --------------------------------------------------------------------------------------------
# verifier


@pytest.fixture
def sector_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ODDBALL_DATA", str(tmp_path))
    sdir = tmp_path / "tess-spoc" / f"s{SECTOR:04d}"
    w = ingest.ShardWriter(sdir, SECTOR, shard_size=2)
    for tic in (1, 2, 3):
        w.add(*convert_fits_bytes(fake_fits(tic)))
    w.flush()
    run = {"started_utc": "2026-11-09T00:00:00Z", "bin_minutes": 30.0, "quality_bitmask": 17087}
    ingest.finalize(SECTOR, "script-url", 5, run=run)
    return sdir


def test_verify_ok(sector_dir):
    rep = verify_sector_dir(sector_dir, SECTOR)
    assert rep["ok"], rep["problems"]
    assert rep["n_rows"] == 3 and rep["n_shards"] == 2 and rep["n_in_script"] == 5


def test_verify_wrong_sector(sector_dir):
    assert not verify_sector_dir(sector_dir, SECTOR + 1)["ok"]


def test_verify_truncated_shard(sector_dir):
    f = sector_dir / "shards" / "lc_00001.parquet"
    f.write_bytes(f.read_bytes()[:-100])
    rep = verify_sector_dir(sector_dir, SECTOR)
    assert not rep["ok"]
    assert any("lc_00001" in p for p in rep["problems"])


def test_verify_missing_shard(sector_dir):
    (sector_dir / "shards" / "lc_00000.parquet").unlink()
    probs = " | ".join(verify_sector_dir(sector_dir, SECTOR)["problems"])
    assert "lc/meta shard indices differ" in probs and "n_converted" in probs


def test_verify_missing_provenance_and_metadata(sector_dir):
    (sector_dir / "provenance.json").unlink()
    (sector_dir / "metadata.parquet").unlink()
    probs = verify_sector_dir(sector_dir, SECTOR)["problems"]
    assert "provenance.json missing" in probs and "metadata.parquet missing" in probs


def test_verify_provenance_count_mismatch(sector_dir):
    p = sector_dir / "provenance.json"
    prov = json.loads(p.read_text())
    prov["n_converted"] += 1
    del prov["doi"]
    p.write_text(json.dumps(prov))
    probs = " | ".join(verify_sector_dir(sector_dir, SECTOR)["problems"])
    assert "n_converted" in probs and "doi" in probs


def test_verify_duplicate_tic_and_stray_files(sector_dir):
    w = ingest.ShardWriter(sector_dir, SECTOR, shard_size=2)  # a second run re-adding TIC 1
    w.add(*convert_fits_bytes(fake_fits(1)))
    w.flush()
    ingest.finalize(SECTOR, "script-url", 5)
    (sector_dir / "x_lc.fits").write_bytes(b"")
    (sector_dir / "shards" / "lc_00009.parquet.part").write_bytes(b"")
    probs = " | ".join(verify_sector_dir(sector_dir, SECTOR)["problems"])
    assert "more than one row" in probs and "stray files" in probs
