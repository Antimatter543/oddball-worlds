"""Survey pilot: target lists, the known-planet reproduction check, and throughput numbers.

  targets   writes OUT/survey_pilot/hosts.csv (11 gate-2 hold-out hosts whose planets gate 2
            recovered blind, 4-43 sectors, mass/radius exactly as the gate-2 host jobs used) and
            OUT/survey_pilot/survey30.csv (30 survey stars, seed 20261112: 12 / 9 / 9 from the
            6-12 / 13-25 / 26+ predicted-sector buckets of survey_targets.csv).
  check     compares the runner's hold-out results with gate 2 (hosts.out.jsonl and
            hosts_short.out.jsonl). Survey-star results are never opened here.
  perf      throughput / memory / disk from progress.jsonl only (no signal fields).

Reproduction tolerance (fixed before the pilot ran):
  * PASS per host: the planet signal gate 2 recovered (holdout_result.csv, its slot in the
    long search) is found again in the same slot with |dP/P| < 1e-5 and |dMES|/MES < 1 %.
  * EXACT expected when the downloaded sectors and products equal gate 2's: the NPZ arrays
    are bitwise equal to gate 2's data/planets_spike/lc/tic<T>.npz, and every signal in both
    ranges (all 6 slots) has the same P, t0 and MES to 1e-9 relative. If MAST now serves
    different sectors, the host is reported as "data changed" and judged on PASS only.

Usage: python -I survey/pilot.py targets | check | perf TAG [TAG ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import LC_DIR, OUT  # noqa: E402

G2 = OUT / "gate2"
PILOT = OUT / "survey_pilot"
HOST_TICS = [166184428, 12421862, 306996324, 153065527, 467179528, 198485881, 307849973,
             231080232, 321669174, 235678745, 271971130]


def targets() -> None:
    PILOT.mkdir(parents=True, exist_ok=True)
    jobs = {j["tic"]: j for j in map(json.loads, open(G2 / "jobs" / "hosts.jsonl"))}
    hr = pd.read_csv(G2 / "holdout_result.csv")
    rows = []
    for t in HOST_TICS:
        j = jobs[t]
        g = [json.loads(x) for x in open(G2 / "hosts.out.jsonl")]
        g = next(r for r in g if r["tic"] == t)
        rows.append({"tic": t, "mass": j["mass"], "rad": j["radius"],
                     "n_sectors": g["n_sectors"], "sectors": " ".join(map(str, g["sectors"])),
                     "planets": "; ".join(hr.loc[(hr["tic"] == t) & (hr["rec"] == True),  # noqa: E712
                                                 "planet"])})
    pd.DataFrame(rows).to_csv(PILOT / "hosts.csv", index=False)
    s = pd.read_csv(G2 / "survey_targets.csv")
    s = s[s["n_sectors"] >= 6]
    rng = np.random.default_rng(20261112)
    pick = []
    for b, n in (("6-12", 12), ("13-25", 9), ("26+", 9)):
        sb = s[s["bucket"] == b]
        pick.append(sb.iloc[rng.choice(len(sb), n, replace=False)])
    p = pd.concat(pick)
    p.to_csv(PILOT / "survey30.csv", index=False)
    print(pd.DataFrame(rows)[["tic", "n_sectors", "planets"]].to_string(index=False))
    print(p.groupby("bucket")["n_sectors"].agg(["size", "sum"]))


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def check(tag: str = "survey_pilot_hosts") -> None:
    res_dir = OUT / tag / "results"
    hr = pd.read_csv(G2 / "holdout_result.csv")
    g_long = {r["tic"]: r for r in map(json.loads, open(G2 / "hosts.out.jsonl"))}
    g_short = {r["tic"]: r for r in map(json.loads, open(G2 / "hosts_short.out.jsonl"))}
    rows = []
    for t in HOST_TICS:
        p = res_dir / f"tic{t}.json"
        if not p.exists():
            rows.append({"tic": t, "verdict": "NOT RUN"})
            continue
        r = json.loads(p.read_text())
        if r["status"] != "ok":
            rows.append({"tic": t, "verdict": f"status {r['status']}"})
            continue
        gl, gs = g_long[t], g_short[t]
        same_data = (r["sectors_found"] == gl["sectors"])
        # bitwise data check against the gate-2 NPZ
        z1 = np.load(r["npz"], allow_pickle=False)
        z0 = np.load(LC_DIR / f"tic{t}.npz", allow_pickle=False)
        keys = [k for k in z0.files if k != "meta"]
        bitwise = same_data and all(np.array_equal(z0[k], z1[k]) for k in keys)
        prod0 = [s["product"] for s in json.loads(str(z0["meta"]))["sectors"]]
        prod1 = [r["products"][str(s)] for s in r["sectors_found"]]
        # every signal, both ranges
        worst = 0.0
        nsig_eq = True
        for rg, g in (("long", gl), ("short", gs)):
            a, b = r["search"][rg]["signals"], g["signals"]
            nsig_eq &= len(a) == len(b)
            for x, y in zip(a, b):
                worst = max(worst, _rel(x["P"], y["P"]), _rel(x["mes"], y["mes"]),
                            abs(x["t0"] - y["t0"]) / y["P"])
        planets = hr[(hr["tic"] == t) & (hr["rec"] == True)]  # noqa: E712
        for _, pl in planets.iterrows():
            slot = int(pl["slot"])
            sig = r["search"]["long"]["signals"]
            got = sig[slot - 1] if len(sig) >= slot else None
            ok = (got is not None and _rel(got["P"], pl["P_det"]) < 1e-5
                  and _rel(got["mes"], pl["mes"]) < 0.01)
            rows.append({"tic": t, "planet": pl["planet"], "n_sec_g2": gl["n_sectors"],
                         "n_sec_now": r["n_sectors_found"], "same_sectors": same_data,
                         "same_products": prod0 == prod1 if same_data else None,
                         "npz_bitwise": bitwise, "slot": slot, "P_g2": pl["P_det"],
                         "P_now": got["P"] if got else None, "MES_g2": pl["mes"],
                         "MES_now": got["mes"] if got else None,
                         "dMES_rel": _rel(got["mes"], pl["mes"]) if got else None,
                         "all_sig_same_count": nsig_eq, "all_sig_worst_rel": worst,
                         "exact": bool(same_data and bitwise and nsig_eq and worst < 1e-9),
                         "verdict": "PASS" if ok else "FAIL"})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    out = OUT / tag / "reproduction.csv"
    df.to_csv(out, index=False)
    print(f"PASS {int((df['verdict'] == 'PASS').sum())} of {len(df)} planets; exact "
          f"{int(df.get('exact', pd.Series(dtype=bool)).fillna(False).sum())}; written {out}")


def routecheck() -> None:
    """The survey's product listing (observation dataURL) against the URIs gate 2 downloaded
    (stored per sector in every gate-2 NPZ). One request at a time."""
    import time

    sys.path.insert(0, str(HERE))
    import download

    download.ERRLOG = lambda r: print("LOG", r)
    download._set_astroquery_ua()
    rows = []
    for p in sorted(LC_DIR.glob("tic*.npz")):
        tic = int(p.stem[3:])
        with np.load(p, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
        old = {int(s["sector"]): (s["product"], s["uri"]) for s in m["sectors"]}
        old_fail = {int(f["sector"]) for f in m.get("failures", [])}
        t0 = time.time()
        try:
            new = download.list_products_obs(tic)
        except Exception as e:  # noqa: BLE001
            rows.append({"tic": tic, "error": repr(e)[:120]})
            continue
        same = {s for s in old if s in new and new[s] == old[s]}
        rows.append({"tic": tic, "n_old": len(old), "n_new": len(new), "same": len(same),
                     "old_not_new": sorted(set(old) - set(new)),
                     "uri_differs": sorted(s for s in old if s in new and new[s] != old[s]),
                     "new_not_old": sorted(set(new) - set(old)),
                     "new_was_failure": sorted((set(new) - set(old)) & old_fail),
                     "q_s": round(time.time() - t0, 2)})
    d = pd.DataFrame(rows)
    d.to_csv(PILOT / "routecheck.csv", index=False)
    bad = d[(d["old_not_new"].map(len) > 0) | (d["uri_differs"].map(len) > 0)] \
        if "old_not_new" in d else d
    print(f"{len(d)} gate-2 stars; sectors old {int(d['n_old'].sum())}, identical "
          f"{int(d['same'].sum())}; stars with a gate-2 sector missing or a different URI: "
          f"{len(bad)}; stars with extra sectors now: "
          f"{int((d['new_not_old'].map(len) > 0).sum())}; errors "
          f"{int(d['error'].notna().sum()) if 'error' in d else 0}; median query "
          f"{d['q_s'].median():.2f} s")
    if len(bad):
        print(bad.to_string(index=False))
    ex = d[d["new_not_old"].map(len) > 0]
    if len(ex):
        print(ex[["tic", "n_old", "n_new", "new_not_old", "new_was_failure"]].to_string(index=False))


def perf(tags: list[str]) -> None:
    rows = []
    for tag in tags:
        prog = {}
        for line in open(OUT / tag / "progress.jsonl"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            prog[r["tic"]] = r
        rows += [dict(r, tag=tag) for r in prog.values() if r["status"] == "ok"]
    d = pd.DataFrame(rows)
    d["bucket"] = pd.cut(d["n_sectors_found"], [0, 5, 12, 25, 99],
                         labels=["<6", "6-12", "13-25", "26+"])
    g = d.groupby("bucket", observed=True).agg(
        n=("tic", "size"), sec_found=("n_sectors_found", "mean"),
        sec_pred=("n_sectors_pred", "mean"), dl_s=("dl_wall_s", "mean"),
        dl_s_per_sector=("dl_wall_s", lambda x: np.nan),
        cpu_s=("search_cpu_s", "mean"), cpu_max=("search_cpu_s", "max"),
        mb_npz=("npz_bytes", lambda x: x.mean() / 1e6), rss_max=("worker_vmhwm_mb", "max"))
    sec = d.groupby("bucket", observed=True).apply(
        lambda x: x["dl_wall_s"].sum() / x["n_sectors_found"].sum(), include_groups=False)
    g["dl_s_per_sector"] = sec
    print(g.round(2).to_string())
    print(json.dumps({"n": len(d), "dl_s_per_sector_all": float(d["dl_wall_s"].sum()
                                                                 / d["n_sectors_found"].sum()),
                      "npz_bytes_per_sector": float(d["npz_bytes"].sum()
                                                    / d["n_sectors_found"].sum()),
                      "found_over_pred": float(d["n_sectors_found"].sum()
                                               / d["n_sectors_pred"].sum())}, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "targets":
        targets()
    elif cmd == "check":
        check(*sys.argv[2:3])
    elif cmd == "routecheck":
        routecheck()
    elif cmd == "perf":
        perf(sys.argv[2:])
