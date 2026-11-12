"""The survey's built-in sensitivity check: which known planets / TOIs / CTOIs on searched stars
did the frozen long search recover, and were the missed ones expected to be detectable?

Objects: the ExoFOP TOI table (local copy, any disposition), CTOIs (the vetting's cached ExoFOP
download), exoplanet.eu confirmed planets within 30" of the survey target (local copy). For each
object on a searched star, with the star's rebuilt survey light curve (`get_lightcurve`) and the
frozen preprocessing (`search2.preprocess2`, rot from gate2/frozen.json):
  n_tr      transits in the data (gate-2 `epochs_in_data`, >= 50% coverage)
  mes_pred  gate-2 prediction: pred_eff x depth x sqrt(sum 1/sigma^2 over in-transit bins), 0 if
            n_tr < 2 (depth: TOI depth, else (R_p / R_*)^2; T14: TOI duration, else circular b=0.3)
  matched   any stored long-search signal (all 6 slots, MES >= 6) that matches by the gate-2 rule
            (`inject.match`: P or an n <= 6 alias within 1%, and >= 80% of epochs line up);
            recovered = a match with MES >= tau_long
Planets with P outside 10-60 d are listed (aliases can still match) but only 10-60 d ones count
for the long-range check. Usage: python -I survey/known_check.py  -> OUT/survey/vet/known.csv
"""

from __future__ import annotations

import json
import math
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
sys.path.insert(0, str(HERE))

from common import CAT_DIR, OUT

RJ_RE = 11.209


def _sep(ra1, de1, ra2, de2):
    r1, d1, r2, d2 = (np.radians(x) for x in (ra1, de1, ra2, de2))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


def _f(x):
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return float(v) if np.isfinite(v) else None


def objects(searched: dict) -> list[dict]:
    tg = pd.read_csv(OUT / "gate2" / "survey_targets.csv")
    tg = tg[tg.tic.isin(set(searched))]
    out = []
    toi = pd.read_csv(CAT_DIR / "toi.csv", low_memory=False)
    for r in toi[toi["TIC ID"].isin(set(searched))].to_dict("records"):
        ep = _f(r.get("Epoch (BJD)"))
        out.append({"tic": int(r["TIC ID"]), "name": f"TOI-{r['TOI']}", "src": "toi",
                    "disp": str(r.get("TFOPWG Disposition")), "P": _f(r.get("Period (days)")),
                    "t0": ep - 2457000 if ep else None, "dur_h": _f(r.get("Duration (hours)")),
                    "depth_ppm": _f(r.get("Depth (ppm)")),
                    "rp_re": _f(r.get("Planet Radius (R_Earth)")), "transit": True,
                    "alias": str(r.get("Planet Name", "")) if str(r.get("Planet Name", ""))
                    != "nan" else ""})
    fp = OUT / "vetting" / "cache" / "ctoi.txt"
    if fp.exists():
        ct = pd.read_csv(StringIO(fp.read_text()), sep="|", low_memory=False)
        for r in ct[ct["TIC ID"].isin(set(searched))].to_dict("records"):
            if r.get("Promoted to TOI") and str(r.get("Promoted to TOI")) not in ("nan", ""):
                continue  # listed under its TOI
            ep = _f(r.get("Transit Epoch (BJD)"))
            out.append({"tic": int(r["TIC ID"]), "name": f"CTOI {r['CTOI']}", "src": "ctoi",
                        "disp": str(r.get("User Disposition")), "P": _f(r.get("Period (days)")),
                        "t0": ep - 2457000 if ep else None,
                        "dur_h": _f(r.get("Duration (hrs)")),
                        "depth_ppm": _f(r.get("Depth ppm")),
                        "rp_re": _f(r.get("Planet Radius (R_Earth)")), "transit": True,
                        "alias": ""})
    else:
        print("WARNING: no CTOI cache; CTOIs not checked")
    eu = pd.read_csv(CAT_DIR / "exoplanet_eu.csv", low_memory=False)
    eu = eu[eu.planet_status == "Confirmed"]
    for s in tg.itertuples():
        sep = _sep(s.ra, s.dec, eu.ra.to_numpy(float), eu.dec.to_numpy(float))
        for j in np.flatnonzero(sep < 30):
            r = eu.iloc[j]
            t0 = _f(r.get("tzero_tr"))
            if t0 and t0 > 2400000:
                t0 -= 2457000
            rp = _f(r.get("radius"))
            out.append({"tic": int(s.tic), "name": str(r["name"]), "src": "exoplanet.eu",
                        "disp": str(r.get("detection_type")), "P": _f(r.get("orbital_period")),
                        "t0": t0, "dur_h": None, "depth_ppm": None,
                        "rp_re": rp * RJ_RE if rp else None,
                        "transit": "Primary Transit" in str(r.get("detection_type")),
                        "alias": ""})
    return out


def main() -> None:
    from get_lightcurve import get_lightcurve
    from inject import duration_14, epochs_in_data, match
    from search2 import preprocess2

    cfg = json.loads((OUT / "gate2" / "frozen.json").read_text())
    tau, eff, rot = cfg["tau_long"], cfg["pred_eff"], cfg["rot"]
    rows = {r["tic"]: r for r in map(json.loads, (OUT / "survey" / "vet" / "rows_long.jsonl").read_text().splitlines())}
    jobs = {j["tic"]: j for j in map(json.loads, (OUT / "survey" / "vet" / "jobs_long.jsonl").read_text().splitlines())}
    objs = objects(rows)
    print(len(objs), "objects on", len({o["tic"] for o in objs}), "searched stars", flush=True)
    res = []
    pp_cache = {}
    for o in sorted(objs, key=lambda o: o["tic"]):
        tic = o["tic"]
        row, job = rows[tic], jobs[tic]
        rec = dict(o, n_sectors=row["n_sectors"])
        if not o["P"] or o["P"] <= 0:
            res.append(dict(rec, note="no period"))
            continue
        if tic not in pp_cache:
            lc, _ = get_lightcurve(tic, 2_000_000_000)
            pp_cache.clear()
            pp_cache[tic] = preprocess2(lc, rot)[0]
        pp = pp_cache[tic]
        M, R = job["mass"], job["radius"]
        dur = o["dur_h"] / 24 if o["dur_h"] else duration_14(o["P"], o["rp_re"] or 1.5, 0.3, M, R)
        depth = (o["depth_ppm"] * 1e-6 if o["depth_ppm"] else
                 ((o["rp_re"] * 0.0091577 / R) ** 2 if o["rp_re"] else None))
        best = None
        for k, sg in enumerate(row["signals"]):
            if o["t0"] is None:
                ratio = sg["P"] / o["P"]
                n = next((c for c in (ratio, 1 / ratio) if 1 <= round(c) <= 6
                          and abs(c - round(c)) <= 0.01 * round(c)), None)
                m = (False, n, None, None)  # period only: not counted as a match
            else:
                m = match(sg, o["P"], o["t0"], dur, pp["idx"])
            if m[0] and (best is None or sg["mes"] > best[1]["mes"]):
                best = (k, sg, m)
            if not m[0] and m[1] is not None and o["t0"] is None:
                rec.setdefault("period_only", []).append((k, round(sg["P"], 4), round(sg["mes"], 2)))
        if o["t0"] is not None:
            n_tr = len(epochs_in_data(pp["idx"], o["P"], o["t0"], dur))
            inb = np.abs((pp["idx"] + 0.5) * (10 / 1440) - o["t0"]
                         - np.round(((pp["idx"] + 0.5) * (10 / 1440) - o["t0"]) / o["P"]) * o["P"]) \
                < dur / 2
            snr = depth * math.sqrt(float(np.sum(1 / pp["sigma"][inb] ** 2))) if depth else None
            rec.update(n_tr=n_tr, mes_pred=(eff * snr if snr and n_tr >= 2 else 0.0)
                       if o["transit"] else None)
        if best:
            k, sg, m = best
            rec.update(match_slot=k, match_P=sg["P"], match_mes=sg["mes"], alias_n=m[1],
                       recovered=sg["mes"] >= tau)
        else:
            rec["recovered"] = False
        res.append(rec)
        print(o["name"], tic, o["P"], rec.get("n_tr"), rec.get("mes_pred"), rec.get("match_mes"),
              rec["recovered"], flush=True)
    df = pd.DataFrame(res)
    df.to_csv(OUT / "survey" / "vet" / "known.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
