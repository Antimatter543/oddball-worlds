"""Build the gate-3 Kaggle kernel folders (nothing is pushed here).

Per shard I of N: <OUT>/gate3/kaggle_kernels/<slug>-IIIofNNN/{gate3_shard.py, kernel-metadata.json}
with the shard's far sample stars, its 50 pc reference stars, their star_info entries and an
embedded `git archive` of --commit (default HEAD; spike code + src/oddball, as the smoke test).
Private, internet on, no GPU (the survey's metadata template). Usage:
  python -I gate3/build3.py [--of 5] [--search-hours 3.2] [--commit SHA]
Then: kaggle kernels push -p <folder>  (or survey/kaggle/shards.sh with TAG/SLUG set)
"""

from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey" / "kaggle"))
from build_kernel import EMBED_PATHS, PINS, PYTHON, UV, git, kaggle_user
from common import OUT

G3 = OUT / "gate3"


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--of", type=int, default=5)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--search-hours", type=float, default=3.2)
    ap.add_argument("--vet-minutes", type=float, default=30.0)
    ap.add_argument("--commit", default=None)
    ap.add_argument("--slug", default="transit-gate3")
    a = ap.parse_args()
    commit = a.commit or git("rev-parse", "HEAD").decode().strip()
    code = base64.b64encode(git("archive", "--format=tar.gz", commit, *EMBED_PATHS)).decode()
    sample = pd.read_csv(G3 / "sample.csv")
    ref = pd.read_csv(G3 / "ref.csv")
    info = json.loads((G3 / "star_info.json").read_text())
    missing = [int(t) for t in sample.tic if str(int(t)) not in info]
    if missing:
        raise SystemExit(f"{len(missing)} sample stars have no star_info entry: {missing[:5]}")
    tmpl = (HERE / "kernel3_template.py").read_text()
    meta_t = json.loads((HERE.parent / "survey" / "kaggle" / "kernel-metadata.template.json")
                        .read_text())
    user = kaggle_user()
    shards = [int(x) for x in a.shards.split(",")] if a.shards else range(a.of)
    for i in shards:
        s = sample[sample.shard % a.of == i]
        r = ref[ref.shard % a.of == i]
        files = {}
        for name, df in (("sample.csv", s), ("ref.csv", r)):
            b = io.StringIO()
            df.to_csv(b, index=False)
            files[name] = b.getvalue()
        files["star_info.json"] = json.dumps({str(int(t)): info[str(int(t))] for t in s.tic})
        slug = f"{a.slug}-{i:03d}of{a.of:03d}"
        cfg = {"shard": i, "of": a.of, "tag": f"gate3_{i:03d}of{a.of:03d}", "commit": commit,
               "pins": PINS, "python": PYTHON, "uv": UV, "search_hours": a.search_hours,
               "vet_minutes": a.vet_minutes, "nonce": secrets.token_hex(4),
               "n_far": len(s), "n_ref": len(r)}
        src = (tmpl.replace("@@CFG@@", base64.b64encode(json.dumps(cfg).encode()).decode())
               .replace("@@FILES@@", base64.b64encode(gzip.compress(
                   json.dumps(files).encode(), 9)).decode())
               .replace("@@CODE@@", code))
        d = G3 / "kaggle_kernels" / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "gate3_shard.py").write_text(src)
        meta = dict(meta_t, id=f"{user}/{slug}", title=slug.replace("-", " "),
                    code_file="gate3_shard.py")
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        (d / "cfg.json").write_text(json.dumps(cfg, indent=1))
        print(f"{d}  ({len(src) / 1e3:.0f} kB, {len(s)} far + {len(r)} ref, commit {commit[:12]}, "
              f"nonce {cfg['nonce']})")


if __name__ == "__main__":
    main()
