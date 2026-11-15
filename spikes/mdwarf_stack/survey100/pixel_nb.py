"""Neighbour-resolved pixel diagnostic for 100 pc survivors (research, not a vetting test; the
frozen pixel test's result stands in the tier).

The frozen pixel test compares the target with ONE neighbour: the brightest Gaia source 1.5-3 px
away. These stars are faint and crowded, so a brighter star closer than 1.5 px, or a second
neighbour, can be missed. Here, per epoch (the frozen test's sectors and difference image,
oddball.gallery.pixels.analyse), the 3x3 difference-image SNR is measured at the target and at
EVERY Gaia neighbour within 3 px (63"), together with the difference-image centroid and the Gaia
source nearest to it. For each neighbour the eclipse depth it would need to make the dip is
depth x 10^(0.4 dG) (G band, aperture effects ignored). Combined SNR = sum / sqrt(n).

Writes <OUT>/survey100/vet/pixel_nb.jsonl. Usage: python -I survey100/pixel_nb.py KEY [KEY ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "vetting"))
from common100 import VET

PX = 21.0


def run(r, star, max_sectors=4):
    from pixel_check import _box_snr

    from oddball.gallery.pixels import analyse, fetch_cube, wcs_of

    s = r["stats"]
    g = star["gaia"]
    ra, dec = g["target"]["ra"], g["target"]["dec"]
    nbs = [n for n in g["neighbours"] if n["sep"] <= 3 * PX]
    # star_info's neighbour lists come from one batch Gaia query whose rows are assigned to the
    # nearest vetted target, so a neighbour that is itself a vetted target is missing there. The
    # per-star research cone (63", every row) is used instead when it exists.
    rp = VET / "research" / f"{r['tic']}.json"
    if rp.exists():
        d = json.loads(rp.read_text())
        if d.get("gaia", {}).get("status") == "ok":
            rows = d["gaia"]["data"]["rows"]
            tg = min(rows, key=lambda x: x["_r"])
            nbs = [{"source": str(x["Source"]), "ra": x["RA_ICRS"], "dec": x["DE_ICRS"],
                    "G": x["Gmag"], "dG": x["Gmag"] - tg["Gmag"], "sep": 3600 * float(np.hypot(
                        (x["RA_ICRS"] - tg["RA_ICRS"]) * np.cos(np.radians(tg["DE_ICRS"])),
                        x["DE_ICRS"] - tg["DE_ICRS"]))}
                   for x in rows if x is not tg and x.get("Gmag") is not None]
            nbs = sorted([n for n in nbs if n["sep"] <= 3 * PX], key=lambda n: n["dG"])
    ses = np.array(s["epoch_ses"])
    secs = np.array(s["epoch_sector"])
    chosen = []
    for j in np.argsort(-ses):
        if int(secs[j]) not in chosen:
            chosen.append(int(secs[j]))
        if len(chosen) >= max_sectors:
            break
    per = []
    for sec in chosen:
        try:
            cube = fetch_cube(ra, dec, sec)
        except Exception as e:  # noqa: BLE001
            per.append({"sector": sec, "error": repr(e)[:150]})
            continue
        w = wcs_of(cube)
        for t, e_s, e_sec in zip(s["epochs"], ses, secs):
            if int(e_sec) != sec:
                continue
            res = analyse(cube, ra, dec, t, s["dur_h"] / 24, "dip")
            if "diff_snr" not in res:
                continue
            d = {"sector": sec, "t": t, "ses": float(e_s), "tgt": res["diff_target3x3_snr"],
                 "peak": res["diff_peak_snr"], "off_px": res["diff_offset_px"], "nb": {}}
            cra, cdec = (float(v) for v in w.pixel_to_world_values(res["diff_cx"], res["diff_cy"]))
            d["centroid_radec"] = [cra, cdec]
            for n in nbs:
                x, y = (float(v) for v in w.world_to_pixel_values(n["ra"], n["dec"]))
                d["nb"][n["source"]] = _box_snr(res["diff_img"], res["diff_snr"], round(y), round(x))
            srcs = [("target", ra, dec)] + [(n["source"], n["ra"], n["dec"]) for n in nbs]
            dd = [(nm, 3600 * np.hypot((a - cra) * np.cos(np.radians(dec)), b - cdec))
                  for nm, a, b in srcs]
            d["nearest_to_centroid"] = min(dd, key=lambda x: x[1])
            per.append(d)
        del cube
        time.sleep(1.0)
    ok = [p for p in per if "tgt" in p]
    n = len(ok)
    out = {"key": r["key"], "tic": r["tic"], "n_epochs": n, "per_epoch": per}
    if n:
        out["S_t"] = float(np.nansum([p["tgt"] for p in ok]) / np.sqrt(n))
        out["neighbours"] = []
        for nb in nbs:
            v = [p["nb"].get(nb["source"]) for p in ok]
            v = [x for x in v if x is not None and np.isfinite(x)]
            out["neighbours"].append({
                "source": nb["source"], "sep": nb["sep"], "dG": nb["dG"], "G": nb["G"],
                "S": float(np.sum(v) / np.sqrt(len(v))) if v else None,
                "need_depth": float(s["depth"] * 10 ** (0.4 * nb["dG"]))})
        out["centroid_nearest"] = [p["nearest_to_centroid"] for p in ok]
    return out


def main():
    rows = {json.loads(x)["key"]: json.loads(x) for x in open(VET / "graded.jsonl") if x.strip()}
    info = json.loads((VET / "star_info.json").read_text())
    outp = VET / "pixel_nb.jsonl"
    done = {json.loads(x)["key"] for x in open(outp)} if outp.exists() else set()
    with open(outp, "a") as fh:
        for k in sys.argv[1:]:
            if k in done:
                continue
            r = rows[k]
            try:
                res = run(r, info[str(r["tic"])])
            except Exception as e:  # noqa: BLE001
                res = {"key": k, "tic": r["tic"], "error": repr(e)[:200]}
            fh.write(json.dumps(res, default=float) + "\n")
            fh.flush()
            nbtxt = "; ".join(f"{n['sep']:.0f}\" dG {n['dG']:.1f} S {n['S'] if n['S'] is None else round(n['S'], 1)}"
                              f" need {100 * n['need_depth']:.1f}%" for n in res.get("neighbours", [])[:6])
            print(k, "S_t", round(res.get("S_t", float("nan")), 1), "|", nbtxt, "| centroid nearest",
                  [(a, round(b, 1)) for a, b in res.get("centroid_nearest", [])], flush=True)


if __name__ == "__main__":
    main()
