"""Merge survey shards into one result file, checking duplicates and gaps. Counts only.

Reads OUT/<tag>/shards/shard<i>of<N>/ (local runs, or Kaggle outputs placed there by
kaggle/fetch_outputs.py). Checks, all printed as counts:
  * every shard 0..N-1 is present, and all manifests agree on frozen_sha256, code_sha256,
    max_sig, rot, min_sectors and targets_sha256;
  * each shard's star list equals the deterministic split of the target list (shard_tics_sha256
    recomputed here from --targets), so no star was assigned twice or never;
  * every result belongs to its shard, carries the manifest's frozen hash, and no TIC has two
    results (duplicates) across shards;
  * gaps: targets with no result, per shard.
Writes OUT/<tag>/merged/results.jsonl.gz (one record per star, sorted by TIC, complete records
for the pre-registered vetting) and merge_report.json (counts, gap and duplicate TIC lists).
Nothing about signals is printed.

Usage: python -I survey/merge.py --of N [--tag survey] [--targets CSV] [--allow-gaps]
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE))

from common import OUT  # noqa: E402

AGREE = ("frozen_sha256", "code_sha256", "max_sig", "rot", "min_sectors", "targets_sha256", "of")


def main() -> int:
    from run_survey import DEFAULT_TARGETS, load_targets, shard_dir, shard_of

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="survey")
    ap.add_argument("--of", type=int, required=True)
    ap.add_argument("--targets", default=str(DEFAULT_TARGETS))
    ap.add_argument("--min-sectors", type=int, default=6)
    ap.add_argument("--allow-gaps", action="store_true", help="write the merge even with gaps")
    a = ap.parse_args()
    base = OUT / a.tag
    tg = load_targets(Path(a.targets), a.min_sectors)
    problems = []
    mans, recs, owner = {}, {}, collections.defaultdict(list)
    gaps = {}
    for i in range(a.of):
        d = shard_dir(base, i, a.of)
        mp = d / "manifest.json"
        if not mp.exists():
            problems.append(f"shard {i}: missing ({d})")
            gaps[i] = [int(t) for t in shard_of(tg, i, a.of)["_tic"]]
            continue
        m = json.loads(mp.read_text())
        mans[i] = m
        want = shard_of(tg, i, a.of)
        sha = hashlib.sha256(",".join(map(str, want["_tic"])).encode()).hexdigest()
        if m.get("shard_tics_sha256") != sha or m.get("shard") != i:
            problems.append(f"shard {i}: star list differs from the deterministic split")
        want_set = set(int(t) for t in want["_tic"])
        got = set()
        for f in sorted((d / "results").glob("tic*.json")):
            r = json.loads(f.read_text())
            t = int(r["tic"])
            if t not in want_set:
                problems.append(f"shard {i}: TIC {t} does not belong to this shard")
            if r.get("frozen_sha256") != m["frozen_sha256"]:
                problems.append(f"shard {i}: TIC {t} has a different frozen hash")
            owner[t].append(i)
            recs[t] = r | {"shard": i}
            got.add(t)
        gaps[i] = sorted(want_set - got)
    ref = next(iter(mans.values()), {})
    for i, m in mans.items():
        bad = [k for k in AGREE if m.get(k) != ref.get(k)]
        if bad:
            problems.append(f"shard {i}: manifest differs from shard {min(mans)} on {bad}")
    dups = {t: s for t, s in owner.items() if len(s) > 1}
    st = collections.Counter(r["status"] for r in recs.values())
    n_gap = sum(len(v) for v in gaps.values())
    report = {"tag": a.tag, "of": a.of, "targets": len(tg), "results": len(recs),
              "status": dict(st), "incomplete": sum(bool(r.get("incomplete")) for r in recs.values()),
              "duplicates": {str(t): s for t, s in dups.items()}, "n_gaps": n_gap,
              "gaps": {str(i): v for i, v in gaps.items() if v}, "problems": problems,
              "frozen_sha256": ref.get("frozen_sha256")}
    md = base / "merged"
    md.mkdir(parents=True, exist_ok=True)
    (md / "merge_report.json").write_text(json.dumps(report, indent=1))
    print(f"targets {len(tg)}  results {len(recs)}  status {dict(st)}  duplicates {len(dups)}  "
          f"gaps {n_gap} " + " ".join(f"s{i}:{len(v)}" for i, v in gaps.items() if v))
    for p in problems[:20]:
        print("PROBLEM", p)
    if problems or dups or (n_gap and not a.allow_gaps):
        print(f"not merged (see {md / 'merge_report.json'}); --allow-gaps writes a partial merge")
        return 1
    tmp = md / "results.jsonl.gz.tmp"
    with gzip.open(tmp, "wt") as fh:
        for t in sorted(recs):
            fh.write(json.dumps(recs[t]) + "\n")
    tmp.replace(md / "results.jsonl.gz")
    print(f"written {md / 'results.jsonl.gz'} ({(md / 'results.jsonl.gz').stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
