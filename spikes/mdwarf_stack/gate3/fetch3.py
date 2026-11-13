"""Pull finished gate-3 kernel outputs into <OUT>/gate3/kaggle_out/<slug>/ (unpacked run/ dir).
Uses the survey fetcher's Kaggle API helper (Content-Length workaround); a fresh temporary
directory per kernel; the kernel's config nonce must match the local build (cfg.json).
Usage: python -I gate3/fetch3.py [--of 5] [--shards 0,1] [--slug transit-gate3]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "survey" / "kaggle"))
from common import OUT
from fetch_outputs import _api

G3 = OUT / "gate3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--of", type=int, default=5)
    ap.add_argument("--shards", default=None)
    ap.add_argument("--slug", default="transit-gate3")
    a = ap.parse_args()
    api = _api()
    user = api.get_config_value("username")
    for i in ([int(x) for x in a.shards.split(",")] if a.shards else range(a.of)):
        slug = f"{a.slug}-{i:03d}of{a.of:03d}"
        cfg = json.loads((G3 / "kaggle_kernels" / slug / "cfg.json").read_text())
        tmp = Path(tempfile.mkdtemp(prefix="g3-kaggle-"))
        try:
            api.kernels_output(f"{user}/{slug}", path=str(tmp), quiet=True)
            tgz = next(tmp.rglob(f"{cfg['tag']}.tar.gz"), None)
            if tgz is None:
                print(f"{slug}: no {cfg['tag']}.tar.gz yet")
                continue
            dst = G3 / "kaggle_out" / slug
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            with tarfile.open(tgz) as t:
                t.extractall(dst, filter="data")
            s = json.loads((dst / "run" / "kaggle_summary.json").read_text())
            ok = s.get("nonce") == cfg["nonce"]
            for f in tmp.rglob("*.log"):
                shutil.copy2(f, dst / f.name)
            print(f"{slug}: unpacked; nonce {'ok' if ok else 'MISMATCH'}; driver exit "
                  f"{s.get('driver_exit')}; {s.get('hours', 0):.2f} h")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
