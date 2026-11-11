"""Write the gate-2 frozen-method entry (outputs/planets_spike/gate2/frozen.json) from DEV only.

  fast   write the fast/slow rule (fixed from the DEV rotation census, before the arm choice)
  arm    apply the pre-registered rotation-arm rule on the DEV paired injections; fix pred_eff
  tau    tau_long (and tau_short when its nulls are complete) from the DEV nulls of the chosen arm

Each step refuses to run if any TEST-star or host search output exists, so nothing here can be
informed by TEST or host results.

  tau_late  fill a pending tau only (mechanical; allowed after TEST/host runs started)

Usage: python -I gate2/freeze.py fast|arm|tau|tau_late
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
import analyze2 as A
from census import POW_STRONG

FZ = A.G2 / "frozen.json"
FAST_MIN_FRAC = 0.5  # fixed 2026-11-11 15:20 before the DEV census table and the arm rule were run
GUARD = ("test_real", "test_real_short", "test_inj", "test_inj_none", "hosts", "hosts_short")


def guard():
    seen = [n for n in GUARD if (A.G2 / f"{n}.out.jsonl").exists()]
    if seen:
        raise SystemExit(f"refusing: TEST/host outputs exist: {seen}")


def load():
    return json.loads(FZ.read_text()) if FZ.exists() else {}


def save(fz):
    FZ.write_text(json.dumps(fz, indent=1))
    print(json.dumps(fz, indent=1))


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def fast():
    guard()
    fz = load()
    fz.update(fast_pow=POW_STRONG, fast_min_frac=FAST_MIN_FRAC, fast_rule_time=now())
    c = A.census()
    dev = [t for t, v in c.items() if v["set"] == "DEV"]
    nf = sum(A.is_fast(t, fz) for t in dev)
    fz["dev_n_fast"] = [int(nf), len(dev)]
    save(fz)


def arm():
    guard()
    fz = load()
    rows = A.ok(A.rd("dev_paired"))
    pairs = {}
    for r in rows:
        if r["id"].startswith("devinj"):
            pairs.setdefault(r["id"].rsplit("-", 1)[0], {})[r["rot"]] = r
    pairs = {k: v for k, v in pairs.items() if len(v) == 2}
    res = {}
    for grp in ("slow", "fast"):
        ks = [k for k, v in pairs.items() if A.is_fast(v["harm"]["tic"], fz) == (grp == "fast")]
        kh = sum(A.rec_top(pairs[k]["harm"], A.PROVISIONAL) for k in ks)
        kn = sum(A.rec_top(pairs[k]["none"], A.PROVISIONAL) for k in ks)
        res[grp] = {"n": len(ks), "harm": int(kh), "none": int(kn),
                    "diff": (kh - kn) / max(len(ks), 1)}
    # pre-registered rule; "loses <= 3 points" read as a recovery-fraction drop of <= 0.03
    choose = res["slow"]["diff"] >= -0.03 and res["fast"]["harm"] >= res["fast"]["none"]
    fz.update(rot="harm" if choose else "none", arm_rule=res, arm_rule_mes=A.PROVISIONAL,
              n_pairs=len(pairs), arm_time=now())
    # prediction efficiency: DEV harm-arm injections, SNR_white >= 12, matched (pre-registered)
    h = [r for r in rows if r["id"].startswith("devinj") and r["rot"] == "harm"
         and r["inj"]["snr_white"] >= 12]
    m = [r for r in h if r["inj_match"] and r["inj_match"][0][0]]
    ratio = [r["signals"][0]["mes"] / r["inj"]["snr_white"] for r in m]
    fz.update(pred_eff=float(np.median(ratio)), pred_eff_n=[len(m), len(h)],
              pred_eff_16_84=[float(np.percentile(ratio, 16)), float(np.percentile(ratio, 84))])
    save(fz)


def tau(late=False):
    """late=True (after TEST/host runs started): only fills a tau still marked pending, by the
    same mechanical rule from the same fixed DEV job file; nothing else in the entry changes."""
    if not late:
        guard()
    fz = load()
    nl, ns = A.arm_files(fz)
    for rg, name, need in (("long", nl, 400), ("short", ns, 300)):
        if late and not str(fz.get(f"tau_{rg}_status", "")).startswith("pending"):
            continue
        null = [r for r in A.ok(A.rd(name)) if r["id"].startswith("null")]
        njobs = sum(1 for x in open(A.G2 / "jobs" / f"{name}.jsonl")
                    if x.strip() and json.loads(x)["id"].startswith("null"))
        st = np.sort([A.top(r) for r in null])[::-1]
        if len(null) < njobs:
            fz[f"tau_{rg}"] = None
            fz[f"tau_{rg}_status"] = (f"pending: {len(null)}/{njobs} DEV nulls finished; computed "
                                      "later by the same rule from the same fixed job file")
            if rg == "long" and len(null) < need:
                fz["tau_long_status"] += "; fewer than 400 -> criterion (a) fails"
            continue
        k = int(math.floor(0.005 * len(st)))
        fz[f"tau_{rg}"] = float(st[k])
        fz[f"tau_{rg}_rank"] = k + 1
        fz[f"n_null_{rg}"] = len(null)
        fz[f"tau_{rg}_status"] = "final"
    fz["tau_late_time" if late else "tau_time"] = now()
    save(fz)


if __name__ == "__main__":
    {"fast": fast, "arm": arm, "tau": tau, "tau_late": lambda: tau(late=True)}[sys.argv[1]]()
