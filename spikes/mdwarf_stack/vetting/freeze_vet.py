"""Freeze the vetting thresholds from DEV statistics (pre-registered rule, mechanical).

For each thresholded statistic, on the DEV true signals where it applies:
  soft: if > 3% exceed the default, loosen to the 97% value (rounded up, 3 significant digits);
  fail: if > 1% exceed the default, loosen to max(99% value, soft).
Never tightened. Writes <OUT>/vetting/frozen_vet.json (and a copy next to this file to commit).
Refuses to run once any TEST or hold-out vetting output exists.

Usage: python -I vetting/freeze_vet.py
"""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import OUT
from vet import DEFAULT_THR, applicable, stat_value

V = OUT / "vetting"


def _ceil3(x):
    if x == 0 or not np.isfinite(x):
        return float(x)
    e = math.floor(math.log10(abs(x))) - 2
    return float(math.ceil(x / 10**e) * 10**e)


def main():
    for f in ("vet_test.jsonl", "vet_holdout.jsonl"):
        if (V / f).exists():
            sys.exit(f"refusing: {f} exists, thresholds can no longer be chosen")
    rows = [json.loads(x) for x in open(V / "vet_dev.jsonl") if x.strip()]
    rows = [r for r in rows if "stats" in r]
    true = [r for r in rows if r["pop"] == "dev_true"]
    fa = [r for r in rows if r["pop"].startswith("dev_fa")]
    thr, report = {}, {}
    for key, (soft, fail) in DEFAULT_THR.items():
        test = key.split(".")[0]
        v = np.array([x for x in (stat_value(r["stats"], key) for r in true
                                  if applicable(test, r["stats"]["mode"])) if x is not None])
        s_new, f_new, why = soft, fail, []
        if len(v):
            if np.mean(v > soft) > 0.03:
                s_new = max(soft, _ceil3(float(np.quantile(v, 0.97, method="higher"))))
                why.append(f"soft loosened: {np.mean(v > soft):.3f} of DEV true > default")
            if fail is not None:
                f_new = max(fail, s_new)
                if np.mean(v > fail) > 0.01:
                    f_new = max(fail, s_new, _ceil3(float(np.quantile(v, 0.99, method="higher"))))
                    why.append(f"fail loosened: {np.mean(v > fail):.3f} of DEV true > default")
        thr[key] = (s_new, f_new)
        vf = np.array([x for x in (stat_value(r["stats"], key) for r in fa
                                   if applicable(test, r["stats"]["mode"])) if x is not None])
        report[key] = {
            "default": [soft, fail], "frozen": [s_new, f_new], "why": why,
            "dev_true_n": len(v),
            "dev_true_soft_or_worse": float(np.mean(v > s_new)) if len(v) else None,
            "dev_true_fail": float(np.mean(v > f_new)) if len(v) and f_new is not None else None,
            "dev_fa_n": len(vf),
            "dev_fa_soft_or_worse": float(np.mean(vf > s_new)) if len(vf) else None,
            "dev_fa_fail": float(np.mean(vf > f_new)) if len(vf) and f_new is not None else None,
        }
    out = {"thresholds": thr, "report": report,
           "n_dev_true": len(true), "n_dev_fa": len(fa),
           "frozen_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "rule": "loosen-only: soft to the 97% value if >3% of DEV true exceed it; fail to "
                   "max(99% value, soft) if >1% exceed it"}
    (V / "frozen_vet.json").write_text(json.dumps(out, indent=1))
    (HERE / "frozen_vet.json").write_text(json.dumps(out, indent=1))
    for k, r in report.items():
        print(f"{k:34s} default {r['default']} -> {r['frozen']}  true n={r['dev_true_n']} "
              f"soft+={r['dev_true_soft_or_worse']}  FA n={r['dev_fa_n']} "
              f"soft+={r['dev_fa_soft_or_worse']} fail={r['dev_fa_fail']}")


def load_frozen():
    p = V / "frozen_vet.json"
    if not p.exists():
        p = HERE / "frozen_vet.json"
    d = json.loads(p.read_text())
    return {k: tuple(v) for k, v in d["thresholds"].items()}


if __name__ == "__main__":
    main()
