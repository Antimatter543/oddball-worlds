"""Freeze the v2 thresholds from DEV statistics only (pre-registered rule, mechanical).

Kept v1 tests: v1's frozen thresholds, copied unchanged from frozen_vet.json.
New statistics (gap_edge, red_noise, low_margin): the v1 loosen-only rule on the DEV true signals
  soft: if > 3% exceed the default, loosen to the 97% value (rounded up, 3 significant digits);
  fail: if > 1% exceed the default, loosen to max(99% value, soft). Never tightened.
  low_margin: denominator = DEV true signals with MES >= tau_long (tau_det = tau_long); signals
  outside the few-transit class count as passes.
DEV false-alarm rates (MES >= 6 with tau_det 6, MES >= 7 with tau_det 7) are reported, not used.
Refuses to run once any validation vetting output exists.

Usage: python -I vetting/freeze_vet2.py  -> <OUT>/vetting/v2/frozen_vet2.json (+ copy here)
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common import OUT
from freeze_vet import _ceil3, load_frozen
from vet import grade
from vet2 import NEW_DEFAULT, V1_DROPPED, V1_KEPT, applicable2, grade2, stat2

V = OUT / "vetting"
V2 = V / "v2"
TAU = json.loads((OUT / "gate2" / "frozen.json").read_text())["tau_long"]


def dev_rows():
    v1 = {r["key"]: r for r in map(json.loads, open(V / "vet_dev.jsonl")) if "stats" in r}
    ex = {r["key"]: r for r in map(json.loads, open(V2 / "vet2_dev.jsonl")) if "v2" in r}
    rows = []
    for k, r in v1.items():
        if k in ex:
            r["stats"]["v2"] = ex[k]["v2"]
            rows.append(r)
    return rows, len(v1), len(ex)


def main():
    if list(V2.glob("vet2_val*.jsonl")):
        sys.exit("refusing: validation vetting output exists, thresholds can no longer be chosen")
    rows, n1, n2 = dev_rows()
    true = [r for r in rows if r["pop"] == "dev_true"]
    fa = [r for r in rows if r["pop"].startswith("dev_fa")]
    v1thr = load_frozen()
    thr = {k: tuple(v1thr[k]) for k in V1_KEPT}
    report = {}
    for key, (soft, fail) in NEW_DEFAULT.items():
        test = key.split(".")[0]
        pool = true if test != "low_margin" else [r for r in true if r["stats"]["mes"] >= TAU]
        v = np.array([stat2(r["stats"], key, TAU) for r in pool], float)
        v = v[~np.isnan(v)]
        s_new, f_new, why = soft, fail, []
        if len(v):
            if np.mean(v > soft) > 0.03:
                q = float(np.quantile(v, 0.97, method="higher"))
                s_new = max(soft, _ceil3(q))
                why.append(f"soft loosened: {np.mean(v > soft):.3f} of DEV true > default")
            if fail is not None:
                f_new = max(fail, s_new)
                if np.mean(v > fail) > 0.01:
                    q = float(np.quantile(v, 0.99, method="higher"))
                    f_new = max(fail, s_new, _ceil3(q))
                    why.append(f"fail loosened: {np.mean(v > fail):.3f} of DEV true > default")
        thr[key] = (s_new, f_new)
        rep = {"default": [soft, fail], "frozen": [s_new, f_new], "why": why,
               "dev_true_n": len(v), "dev_true_soft_or_worse": float(np.mean(v > s_new))}
        for lab, mes_min in (("fa6", 6.0), ("fa7", 7.0)):
            sel = [r for r in fa if r["stats"]["mes"] >= mes_min
                   and applicable2(test, r["stats"]["mode"])]
            vf = np.array([stat2(r["stats"], key, mes_min) for r in sel], float)
            vf = vf[~np.isnan(vf)]
            rep[f"dev_{lab}_n"] = len(vf)
            rep[f"dev_{lab}_soft_or_worse"] = float(np.mean(vf > s_new)) if len(vf) else None
            rep[f"dev_{lab}_fail"] = (float(np.mean(vf > f_new)) if len(vf) and f_new is not None
                                      else None)
        report[key] = rep
    # DEV tiers, v1 vs v2 (information; the thresholds above are already fixed)
    tiers = {}
    for lab, sel, tau_det in (
            ("dev_true (MES >= tau)", [r for r in true if r["stats"]["mes"] >= TAU], TAU),
            ("dev_fa >= 7 (tau_det 7)", [r for r in fa if r["stats"]["mes"] >= 7], 7.0),
            ("dev_fa_real >= 6 (tau_det 6)", [r for r in fa if r["pop"] == "dev_fa_real"], 6.0),
            ("dev_fa_null >= 6 (tau_det 6)", [r for r in fa if r["pop"] == "dev_fa_null"], 6.0)):
        tiers[lab] = {"n": len(sel),
                      "v1": dict(Counter(grade(r["stats"], v1thr)["tier"] for r in sel)),
                      "v2": dict(Counter(grade2(r["stats"], thr, tau_det)["tier"] for r in sel))}
    out = {"thresholds": thr, "report": report, "dev_tiers": tiers, "tau_long": TAU,
           "v1_kept": list(V1_KEPT), "v1_dropped": list(V1_DROPPED),
           "n_dev_rows": len(rows), "n_v1_rows": n1, "n_v2_rows": n2,
           "frozen_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
           "rule": "v1 kept tests: v1 frozen thresholds; new: loosen-only on DEV true (soft 97% if "
                   ">3% exceed, fail max(99%, soft) if >1% exceed); low_margin on DEV true MES >= "
                   "tau_long"}
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "frozen_vet2.json").write_text(json.dumps(out, indent=1))
    (HERE / "frozen_vet2.json").write_text(json.dumps(out, indent=1))
    print(f"DEV rows {len(rows)} (v1 {n1}, v2 extras {n2}); true {len(true)}, FA {len(fa)}")
    for k, r in report.items():
        print(f"{k:24s} default {r['default']} -> {r['frozen']}  {r['why']}\n"
              f"    true n={r['dev_true_n']} soft+={r['dev_true_soft_or_worse']:.3f}; "
              f"FA>=6 n={r['dev_fa6_n']} soft+={r['dev_fa6_soft_or_worse']} "
              f"fail={r['dev_fa6_fail']}; FA>=7 n={r['dev_fa7_n']} "
              f"soft+={r['dev_fa7_soft_or_worse']} fail={r['dev_fa7_fail']}")
    for k, v in tiers.items():
        print(f"{k:30s} n={v['n']}  v1 {v['v1']}  v2 {v['v2']}")


if __name__ == "__main__":
    main()
