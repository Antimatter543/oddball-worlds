"""Pull finished shard kernels' outputs into OUT/<tag>/shards/ and run the merge check.

Per shard: download the kernel's output (Kaggle API, with the Content-Length workaround) to a
fresh temporary directory, unpack <tag>_shardIIIofNNN.tar.gz, then add its result records to
the local shard directory: a TIC that already has a local result is kept as it is and counted
(a resumed shard must not repeat stars). progress/errors/downloads JSONL lines are appended,
the manifest is copied if missing (else it must agree), and the kernel log is kept.
Counts only are printed. Finally merge.py runs (unless --no-merge).

Usage: python -I survey/kaggle/fetch_outputs.py --of N [--shards 0,1] [--tag survey]
       [--slug transit-survey] [--no-merge]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parent
sys.path.insert(0, str(SURVEY.parent))
sys.path.insert(0, str(SURVEY.parent / "gate2"))
sys.path.insert(0, str(SURVEY))

from common import OUT  # noqa: E402

MUST_AGREE = ("frozen_sha256", "code_sha256", "max_sig", "rot", "shard", "of",
              "shard_tics_sha256", "min_sectors")


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    def patched_download(response, outfile, http_client=None, quiet=False, *a, **k):
        with open(outfile, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 16):
                f.write(chunk)

    api.download_file = patched_download
    return api


def fetch_one(api, user, slug, tag, i, of) -> str:
    from run_survey import shard_dir

    tmp = Path(tempfile.mkdtemp(prefix="ow-kaggle-"))
    try:
        api.kernels_output(f"{user}/{slug}", path=str(tmp), quiet=True)
        name = f"{tag}_shard{i:03d}of{of:03d}.tar.gz"
        tgz = next(tmp.rglob(name), None)
        if tgz is None:
            return f"shard {i}: no {name} in the kernel output (still running or failed?)"
        ex = tmp / "x"
        with tarfile.open(tgz) as t:
            t.extractall(ex, filter="data")
        src = ex / "outputs" / "planets_spike" / tag
        src = src / "shards" / f"shard{i:03d}of{of:03d}" if of > 1 else src
        dst = shard_dir(OUT / tag, i, of)
        (dst / "results").mkdir(parents=True, exist_ok=True)
        m_src = json.loads((src / "manifest.json").read_text())
        m_dst = dst / "manifest.json"
        if m_dst.exists():
            old = json.loads(m_dst.read_text())
            bad = [k for k in MUST_AGREE if old.get(k) != m_src.get(k)]
            if bad:
                return f"shard {i}: manifest disagrees on {bad}; nothing copied"
            old["starts"] = old.get("starts", []) + m_src.get("starts", [])
            m_dst.write_text(json.dumps(old, indent=1))
        else:
            shutil.copy2(src / "manifest.json", m_dst)
        new = kept = 0
        for f in (src / "results").glob("tic*.json"):
            g = dst / "results" / f.name
            if g.exists():
                kept += 1
                continue
            shutil.copy2(f, g)
            new += 1
        for name in ("progress.jsonl", "errors.jsonl", "downloads.jsonl"):
            if (src / name).exists():
                with open(dst / name, "a") as fh:
                    fh.write((src / name).read_text())
        for name in ("survey.log", "kaggle_summary.json"):
            if (src / name).exists():
                (dst / "kaggle").mkdir(exist_ok=True)
                shutil.copy2(src / name, dst / "kaggle" / f"{name}.{len(m_src.get('starts', []))}")
        return f"shard {i}: {new} new result records, {kept} already present"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--tag", default="survey")
    ap.add_argument("--slug", default="transit-survey")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--min-sectors", type=int, default=6)
    ap.add_argument("--targets", default=None)
    a = ap.parse_args()
    api = _api()
    user = api.get_config_value("username")
    shards = [int(x) for x in a.shards.split(",")] if a.shards else range(a.of)
    for i in shards:
        slug = f"{a.slug}-{i:03d}of{a.of:03d}" if a.of > 1 else a.slug
        try:
            print(fetch_one(api, user, slug, a.tag, i, a.of), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"shard {i}: fetch failed {e!r:.200}", flush=True)
    if a.no_merge:
        return 0
    cmd = [sys.executable, "-I", str(SURVEY / "merge.py"), "--tag", a.tag, "--of", str(a.of),
           "--min-sectors", str(a.min_sectors)]
    if a.targets:
        cmd += ["--targets", a.targets]
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
