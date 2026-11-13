"""Part 2.2: transit-timescale noise of alternative products vs the survey's SPOC / TESS-SPOC
light curve, same star and sector.

Sample (seed 20261113, fixed before any noise was measured): survey stars with status ok,
6 per Tmag bin [10,11) .. [14,15) (30 in all); for each, one searched sector <= 98 drawn at
random among those with a QLP file on the public bucket. Products: survey (as searched), QLP
SAP_FLUX, TGLC aperture flux (if MAST lists one), our TESScut aperture photometry.

Metric: robust (MAD) scatter of 2-hour bins of the frozen-preprocessed single-sector light curve
(search2.preprocess2, rot = harm), bins with >= 50% of their expected points; also the number of
usable 10-min bins (duty cycle) and the share of the sector's time span covered.

Writes FU_OUT/noise.jsonl. Usage: python -I followup/noise.py
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fu
import products
from get_lightcurve import get_lightcurve

SEED = 20261113
OUTF = fu.FU_OUT / "noise.jsonl"


def sigma_2h(pp, cad_bins):
    t = (pp["idx"] + 0.5) * fu.BIN_D
    k = np.floor(t * 12).astype(np.int64)
    u, inv = np.unique(k, return_inverse=True)
    n = np.bincount(inv)
    b = np.bincount(inv, weights=pp["flux"]) / n
    need = 0.5 * 12 / cad_bins
    b = b[n >= need]
    if len(b) < 20:
        return float("nan"), len(b)
    return float(1.4826 * np.median(np.abs(b - np.median(b)))), len(b)


def measure(lc):
    pp = fu.prep(lc)
    d = np.diff(pp["idx"])
    cad = max(1, int(np.median(d[d > 0])))
    s2, nb = sigma_2h(pp, cad)
    return {"sigma2h_ppm": 1e6 * s2, "n_2h": nb, "n_bins": len(pp["idx"]),
            "span_d": float((pp["idx"].max() - pp["idx"].min()) * fu.BIN_D),
            "cov_hours": float(len(pp["idx"]) * cad * fu.BIN_D * 24), "cad_min": 10 * cad}


def pick():
    rows = []
    with gzip.open(fu.OUT / "survey" / "merged" / "results.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r["status"] == "ok":
                rows.append((r["tic"], r["products"]))
    tg = pd.read_csv(fu.OUT / "gate2" / "survey_targets.csv").set_index("tic")
    rng = np.random.default_rng(SEED)
    out = []
    for lo in range(10, 15):
        cand = [(t, p) for t, p in rows if t in tg.index and lo <= tg.loc[t, "Tmag"] < lo + 1]
        order = rng.permutation(len(cand))
        got = 0
        for j in order:
            t, p = cand[j]
            secs = [int(s) for s in p if int(s) <= 98]
            rng.shuffle(secs)
            for s in secs:
                try:
                    products._get(products.S3 + products.qlp_key(t, s), timeout=60)
                except FileNotFoundError:
                    continue
                except Exception:  # noqa: BLE001
                    continue
                out.append({"tic": int(t), "sector": s, "tmag": float(tg.loc[t, "Tmag"]),
                            "survey_product": p[str(s)], "ra": float(tg.loc[t, "ra"]),
                            "dec": float(tg.loc[t, "dec"])})
                got += 1
                break
            if got == 6:
                break
    return out


def main():
    sf = fu.FU_OUT / "noise_sample.json"
    if sf.exists():
        sample = json.loads(sf.read_text())
    else:
        sample = pick()
        sf.write_text(json.dumps(sample, indent=1))
    done = set()
    if OUTF.exists():
        done = {(json.loads(x)["tic"], json.loads(x)["product"]) for x in
                OUTF.read_text().splitlines()}
    from astroquery.mast import Catalogs  # (UA and timeout set by mast_hlsp's import)
    from tess_test import mast_hlsp

    gaia = {}
    try:
        import download

        download._set_astroquery_ua()
        cat = Catalogs.query_criteria(catalog="Tic", ID=[str(x["tic"]) for x in sample])
        for i, g in zip(cat["ID"], cat["GAIA"]):
            try:
                gaia[int(i)] = int(g)  # TIC v8 GAIA = DR2 id (equal to DR3 for most stars)
            except (TypeError, ValueError):
                pass
    except Exception as ex:  # noqa: BLE001
        print("TIC Gaia ids not checked:", repr(ex)[:120], flush=True)
    for smp in sample:
        tic, s = smp["tic"], smp["sector"]
        try:
            hl = mast_hlsp(tic)
        except Exception:  # noqa: BLE001
            hl = {}
        for kind in ("survey", "qlp", "tglc", "tars", "tesscut"):
            if (tic, kind) in done:
                continue
            row = dict(smp, product=kind)
            t0 = time.time()
            try:
                if kind == "survey":
                    lc_all, _ = get_lightcurve(tic, max_bytes=150_000_000)
                    m = lc_all["sector"] == s
                    lc = {k: lc_all[k][m] for k in ("idx", "flux", "sector")}
                elif kind == "qlp":
                    lc, info = products.qlp(tic, s, hl.get(("qlp", s)))
                elif kind == "tglc" and ("tglc", s) not in hl:
                    if s > 55 or gaia.get(tic) is None:
                        row["status"] = "not available"
                        raise StopIteration
                    lc, info = products.tglc_s3(gaia[tic], smp["ra"], smp["dec"], s)
                elif kind in ("tglc", "tars"):
                    if (kind, s) not in hl:
                        row["status"] = "not available"
                        raise StopIteration
                    lc, info = products.generic_hlsp(hl[(kind, s)], s, kind)
                else:
                    lc, info = products.tesscut(smp["ra"], smp["dec"], s)
                    row["aperture_px"] = info["aperture_px"]
                row.update(measure(lc))
                row["status"] = "ok"
            except StopIteration:
                pass
            except Exception as ex:  # noqa: BLE001
                row["status"] = "failed"
                row["why"] = f"{type(ex).__name__}: {ex}"[:200]
                row["trace"] = traceback.format_exc()[-400:]
            row["wall_s"] = round(time.time() - t0, 1)
            with open(OUTF, "a") as fh:
                fh.write(json.dumps(row, default=float) + "\n")
            print(tic, s, kind, row.get("status"), row.get("sigma2h_ppm"), flush=True)


if __name__ == "__main__":
    main()
