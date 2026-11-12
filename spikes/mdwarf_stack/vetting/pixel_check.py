"""Pixel-level check of one vetted signal from TESScut FFI cutouts (pre-registered test 13).

Up to 4 sectors (those holding the highest-SES epochs). Per epoch, the difference image
(`oddball.gallery.pixels.analyse`): target 3x3 SNR, and the 3x3 SNR at the brightest Gaia
neighbour 1.5-3 px away (its pixel from the per-sector WCS). Combined S = sum / sqrt(n).
  uninformative (n/a): max(S_t, S_nb) < 3
  fail: S_nb >= 3 and S_nb > S_t + 2           (the dip is on the neighbour)
  soft: centroid > 1 px off target in >= half of the epochs with peak SNR >= 4
  pass: otherwise
Cutouts are downloaded to a temp dir and deleted by `fetch_cube`; nothing is kept but numbers.

Usage: python -I vetting/pixel_check.py VET.jsonl KEY [KEY ...]  -> <OUT>/vetting/pixel.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "src"))
sys.path.insert(0, str(HERE))
from common import OUT  # noqa: E402

V = OUT / "vetting"
PX = 21.0


def _box_snr(diff, snr, r0, c0):
    ny, nx = diff.shape
    if not (0 <= r0 < ny and 0 <= c0 < nx):
        return np.nan
    sl = (slice(max(0, r0 - 1), r0 + 2), slice(max(0, c0 - 1), c0 + 2))
    d = diff[sl]
    with np.errstate(invalid="ignore", divide="ignore"):
        noise = np.where(snr[sl] != 0, d / snr[sl], np.nan)
    ok = np.isfinite(d) & np.isfinite(noise)
    if not ok.any():
        return np.nan
    return float(d[ok].sum() / np.sqrt((noise[ok] ** 2).sum()))


def pixel_check(stats, star, max_sectors=4):
    from oddball.gallery.pixels import analyse, fetch_cube, wcs_of

    g = (star or {}).get("gaia") or {}
    if "target" not in g:
        return {"error": "no Gaia position for the target"}
    ra, dec = g["target"]["ra"], g["target"]["dec"]
    nbs = [n for n in g["neighbours"] if 1.5 * PX <= n["sep"] <= 3 * PX]
    nb = min(nbs, key=lambda n: n["dG"]) if nbs else None
    ses = np.array(stats["epoch_ses"])
    secs = np.array(stats["epoch_sector"])
    order = np.argsort(-ses)
    chosen = []
    for j in order:
        if secs[j] not in chosen:
            chosen.append(int(secs[j]))
        if len(chosen) >= max_sectors:
            break
    Dd = stats["dur_h"] / 24
    per = []
    errs = []
    for s in chosen:
        try:
            cube = fetch_cube(ra, dec, s)
        except Exception as e:  # noqa: BLE001
            errs.append(f"S{s}: {e!r}"[:120])
            continue
        w = wcs_of(cube)
        for e_t, e_s, e_sec in zip(stats["epochs"], ses, secs):
            if e_sec != s:
                continue
            res = analyse(cube, ra, dec, e_t, Dd, "dip")
            if "diff_snr" not in res:
                continue
            d = {"sector": s, "t": e_t, "ses": float(e_s), "tgt": res["diff_target3x3_snr"],
                 "peak": res["diff_peak_snr"], "off": res["diff_offset_px"]}
            if nb is not None:
                x, y = (float(v) for v in w.world_to_pixel_values(nb["ra"], nb["dec"]))
                d["nb"] = _box_snr(res["diff_img"], res["diff_snr"], round(y), round(x))
            per.append(d)
        del cube
        time.sleep(1.0)
    if not per:
        return {"error": "no usable cutout epoch" + (f" ({'; '.join(errs)})" if errs else "")}
    n = len(per)
    St = float(np.nansum([p["tgt"] for p in per]) / np.sqrt(n))
    nbv = [p.get("nb") for p in per if p.get("nb") is not None and np.isfinite(p.get("nb"))]
    Snb = float(np.sum(nbv) / np.sqrt(len(nbv))) if nbv else float("nan")
    strong = [p for p in per if p["peak"] >= 4]
    n_off = sum(p["off"] > 1.0 for p in strong)
    out = {"S_t": St, "S_nb": Snb, "n_epochs": n, "n_strong": len(strong), "n_off": n_off,
           "nb_dG": nb["dG"] if nb else None, "nb_sep": nb["sep"] if nb else None,
           "sectors": chosen, "errors": errs, "per_epoch": per}
    smax = max(St, Snb if np.isfinite(Snb) else -np.inf)
    if np.isfinite(Snb) and Snb >= 3 and Snb > St + 2:
        out.update(status="fail", why=f"the dip is stronger on a neighbour {nb['sep']:.0f}\" away "
                                      f"(SNR {Snb:.1f} vs {St:.1f} on target)")
    elif smax < 3:
        out.update(status="n/a", why=f"uninformative: pixel SNR {St:.1f} on target (< 3)")
    elif strong and n_off >= len(strong) / 2:
        out.update(status="soft", why=f"difference centroid > 1 px off target in {n_off} of "
                                      f"{len(strong)} strong epochs")
    else:
        out.update(status="pass", why=f"dip on target (pixel SNR {St:.1f}"
                                      + (f", neighbour {Snb:.1f})" if np.isfinite(Snb) else ")"))
    return out


def main():
    from catalog import load_info

    src = Path(sys.argv[1])
    keys = sys.argv[2:]
    rows = {json.loads(x)["key"]: json.loads(x) for x in open(src) if x.strip()}
    info = load_info()
    outp = V / "pixel.jsonl"
    done = {json.loads(x)["key"] for x in open(outp)} if outp.exists() else set()
    with open(outp, "a") as fh:
        for k in keys:
            if k in done:
                continue
            r = rows[k]
            t0 = time.time()
            try:
                res = pixel_check(r["stats"], info.get(str(r["tic"])))
            except Exception as e:  # noqa: BLE001
                res = {"error": repr(e)[:200]}
            fh.write(json.dumps({"key": k, "tic": r["tic"], "pixel": res}, default=float) + "\n")
            fh.flush()
            print(k, res.get("status", "error"), res.get("why", res.get("error")),
                  f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
