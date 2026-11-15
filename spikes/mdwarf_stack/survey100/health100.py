"""Survey health for the 50-100 pc extension (run before any signal is looked at).

Reads <OUT>/survey100/merged/results.jsonl.gz, the per-shard manifests and errors, and the
frozen target list. Reports: status counts, real-sector distribution, per-shard consistency (one
git head, one frozen hash, one set of code hashes), errors and incomplete records, mass/radius
fallbacks, products, and the real sensitivity per searched star from the gate-3 far-star
per-bucket detection rates (`<OUT>/gate3/analyze3.txt` section 3.4, numbers copied below).
No search signal is read here except to count records.

Usage: python -I survey100/health100.py  -> <OUT>/survey100/vet/health.txt, health.json
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import OUT

S100 = OUT / "survey100"
TARGETS = OUT / "gate3" / "targets_100pc.csv"

# Gate-3 far-star detection (= kept, all detected were A/B) by real-sector bucket:
# (detected, injected) per cell, from analyze3.txt 3.4.
G3_RATES = {
    "1.0-1.5 R_e, 15-30 d": {"1-3": (0, 164), "4-5": (0, 112), "6-12": (0, 132), "13+": (3, 62)},
    "1.0-1.5 R_e, 30-50 d": {"1-3": (0, 82), "4-5": (0, 56), "6-12": (0, 66), "13+": (0, 31)},
    "1.5-2.0 R_e, 15-50 d": {"1-3": (0, 82), "4-5": (0, 56), "6-12": (1, 66), "13+": (8, 31)},
    "2.0-3.0 R_e, 15-50 d": {"1-3": (0, 82), "4-5": (2, 56), "6-12": (10, 66), "13+": (24, 31)},
}


def bucket(n):
    return "1-3" if n <= 3 else "4-5" if n <= 5 else "6-12" if n <= 12 else "13+"


def main():
    tg = pd.read_csv(TARGETS)
    recs = []
    with gzip.open(S100 / "merged" / "results.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            r.pop("search", None) if r.get("status") != "ok" else None
            s = r.get("search") or {}
            recs.append({
                "tic": r["tic"], "status": r.get("status"), "shard": r.get("shard"),
                "git": r.get("git_head"), "frozen": r.get("frozen_sha256"),
                "code": json.dumps(r.get("code_sha256"), sort_keys=True),
                "n_pred": r.get("n_sectors_pred"), "n_real": r.get("n_sectors_found") or 0,
                "sectors": r.get("sectors_found") or [], "incomplete": bool(r.get("incomplete")),
                "sector_failures": r.get("sector_failures") or [],
                "mr": r.get("mr_source"), "products": r.get("products") or {},
                "n_s3": r.get("n_from_s3") or 0, "n_mast": r.get("n_from_mast") or 0,
                "cpu": r.get("search_cpu_s") or 0.0, "err": r.get("error"),
                "n_long": len((s.get("long") or {}).get("signals") or []),
                "list_undecided": r.get("list_undecided") or [],
            })
    df = pd.DataFrame(recs)
    out, js = [], {}
    P = out.append
    P("# 50-100 pc survey health (before any signal is read)")
    P(f"records {len(df)}; target list {len(tg)}; unique TICs {df.tic.nunique()}; "
      f"in list {df.tic.isin(tg.tic).sum()}; list stars without a record "
      f"{(~tg.tic.isin(df.tic)).sum()}")
    st = Counter(df.status)
    P(f"status {dict(st)}")
    js["status"] = dict(st)
    P(f"git heads {dict(Counter(df.git))}")
    P(f"frozen hashes {dict(Counter(df.frozen))}")
    P(f"code-hash sets {len(set(df.code))}")
    js["git"] = dict(Counter(df.git))
    js["frozen"] = dict(Counter(df.frozen))
    js["n_code_sets"] = len(set(df.code))
    ok = df[df.status == "ok"]
    nd = df[df.status == "no_data"]
    other = df[~df.status.isin(["ok", "no_data"])]
    P(f"other-status records: {len(other)} {other[['tic', 'status', 'err']].to_dict('records')[:10]}")

    # real sectors
    q = np.percentile(ok.n_real, [10, 25, 50, 75, 90])
    P(f"\n## Real sectors per searched star: median {q[2]:.0f} (10/25/75/90%: {q[0]:.0f} / "
      f"{q[1]:.0f} / {q[3]:.0f} / {q[4]:.0f}); mean {ok.n_real.mean():.2f}")
    bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 12), (13, 25), (26, 999)]
    cnt = [int(((ok.n_real >= a) & (ok.n_real <= b)).sum()) for a, b in bins]
    P("searched stars with 1 / 2 / 3 / 4-5 / 6-12 / 13-25 / 26+ real sectors: "
      + " / ".join(f"{c:,}" for c in cnt))
    P(f"fewer than 6 real sectors: {(ok.n_real < 6).sum():,} ({(ok.n_real < 6).mean():.0%})")
    js["real_sector_bins"] = cnt
    pred = df.n_pred.sum()
    P(f"real / predicted sectors: searched stars {ok.n_real.sum() / ok.n_pred.sum():.3f}; "
      f"all stars {df.n_real.sum() / pred:.3f} (gate-3 sample 0.666 / 0.560; 50 pc 0.78)")
    P(f"no-data share {len(nd) / len(df):.3f} (gate-3 sample 0.217; 50 pc 0.071)")
    tmag = tg.set_index("tic").Tmag
    df["tmag"] = df.tic.map(tmag)
    for lo, hi in ((-5, 12), (12, 13), (13, 14), (14, 15.01)):
        sel = df[(df.tmag >= lo) & (df.tmag < hi)]
        if len(sel):
            P(f"  T {lo}-{hi:.0f}: n {len(sel)}, no data {(sel.status == 'no_data').mean():.3f}, "
              f"real/pred {sel.n_real.sum() / sel.n_pred.sum():.3f}")
    allsec = Counter(s for ss in ok.sectors for s in ss)
    P(f"sector range {min(allsec)}-{max(allsec)}; star-sectors {sum(allsec.values()):,}; "
      f"star-sectors from S87+ {sum(v for k, v in allsec.items() if k >= 87):,}")
    prod = Counter(p for d in ok.products for p in d.values())
    P(f"products (star-sectors) {dict(prod)}; bytes from S3 {ok.n_s3.sum():,}, MAST {ok.n_mast.sum():,}")
    js["products"] = dict(prod)

    # per shard
    P("\n## Per shard")
    P("| shard | n | ok | no_data | other | mean real sectors (ok) | CPU h | incomplete | git heads | frozen |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    sh = []
    for s, g in df.groupby("shard"):
        o = g[g.status == "ok"]
        row = dict(shard=int(s), n=len(g), ok=len(o), nd=int((g.status == "no_data").sum()),
                   other=int((~g.status.isin(["ok", "no_data"])).sum()),
                   mean_sec=float(o.n_real.mean()), cpu_h=float(o.cpu.sum() / 3600),
                   inc=int(g.incomplete.sum()), git=g.git.nunique(), fz=g.frozen.nunique())
        sh.append(row)
        P(f"| {s} | {row['n']} | {row['ok']} | {row['nd']} | {row['other']} | {row['mean_sec']:.2f} | "
          f"{row['cpu_h']:.2f} | {row['inc']} | {row['git']} | {row['fz']} |")
    shd = pd.DataFrame(sh)
    P(f"ok share by shard {(shd.ok / shd.n).min():.3f}-{(shd.ok / shd.n).max():.3f}; mean sectors "
      f"{shd.mean_sec.min():.2f}-{shd.mean_sec.max():.2f}; CPU {shd.cpu_h.min():.2f}-{shd.cpu_h.max():.2f} h "
      f"(total {shd.cpu_h.sum():.1f} search CPU-h); z of ok share max "
      f"{np.max(np.abs((shd.ok / shd.n - len(ok) / len(df)) / np.sqrt(len(ok) / len(df) * (1 - len(ok) / len(df)) / shd.n))):.2f}")
    js["shards"] = sh

    # manifests
    mans = []
    for p in sorted((S100 / "shards").glob("shard*/manifest.json")):
        m = json.loads(p.read_text())
        mans.append((m.get("frozen_sha256"), json.dumps(m.get("code_sha256"), sort_keys=True),
                     m.get("targets_sha256"), m.get("n_targets")))
    P(f"manifests {len(mans)}; frozen hashes {len({m[0] for m in mans})}; code-hash sets "
      f"{len({m[1] for m in mans})} (same as records: {mans[0][1] == df.code.iloc[0]}); targets sha "
      f"{len({m[2] for m in mans})} distinct (one per shard file); n_targets sum {sum(m[3] for m in mans)}")

    # errors
    P("\n## Errors and incomplete records")
    ek = Counter()
    final = Counter()
    for p in sorted((S100 / "shards").glob("shard*/errors.jsonl")):
        for x in p.read_text().splitlines():
            if x.strip():
                e = json.loads(x)
                ek[e.get("kind")] += 1
                if e.get("final"):
                    final[e.get("kind")] += 1
    P(f"shard error log lines by kind {dict(ek)}; marked final {dict(final)}")
    js["errors"] = dict(ek)
    js["errors_final"] = dict(final)
    inc = df[df.incomplete]
    P(f"incomplete records {len(inc)} (status {dict(Counter(inc.status))}); sector failures in "
      f"records: {sum(len(x) for x in df.sector_failures)} on {sum(bool(x) for x in df.sector_failures)} stars")
    why = Counter()
    for x in df.sector_failures:
        for f in x:
            why[(f.get("error") if isinstance(f, dict) else str(f))[:70]] += 1
    P(f"sector-failure reasons {dict(why.most_common(8))}")
    und = df[df.list_undecided.map(len) > 0]
    P(f"stars with undecided product listings {len(und)}")
    inc_tics = sorted(inc.tic.tolist())
    P(f"incomplete TICs {inc_tics}")
    js["incomplete"] = inc_tics
    P(f"mass/radius source {dict(Counter(df.mr))}")

    # sensitivity
    P("\n## Real sensitivity per searched star (gate-3 far per-bucket detection rates)")
    bk = Counter(bucket(n) for n in ok.n_real)
    P(f"real-sector buckets of searched stars {dict(bk)}")
    sens = {}
    for cell, rates in G3_RATES.items():
        p = sum(bk[b] * rates[b][0] / rates[b][1] for b in rates) / len(ok)
        sens[cell] = p
        P(f"- {cell}: {p:.4f} detected per searched star with a transiting planet in the cell "
          f"(gate-3 sample: {sum(v[0] for v in rates.values())}/{sum(v[1] for v in rates.values())})")
    js["sensitivity"] = sens
    (S100 / "vet").mkdir(parents=True, exist_ok=True)
    (S100 / "vet" / "health.txt").write_text("\n".join(out) + "\n")
    (S100 / "vet" / "health.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(out))


if __name__ == "__main__":
    main()
