"""Shorter-period aliases P/k (k = 2..5) of a 100 pc survivor, on the survey light curve (frozen
preprocessing). The epochs that P/k adds (T0 + (m + j/k) P, j = 1..k-1) that fall in the data are
combined (joint depth, weights 1/sigma^2 with the per-sector red-noise factor), and compared with
the same joint value after the signal's box is injected at all of them into the RAW flux, at the raw
depth d_raw from rereduce.py when it exists (else the detected depth, which under-states it) (what a
real dip would give after the detrend). Rule as in followup/alias.py (pre-registered in
docs/spikes/mdwarf_followup.md): excluded if the injected copy is recovered (joint SES >= 4) and
the observed joint SES is more than 3 below it. Also prints P / 13.70 d and P / 27.40 d (TESS orbit).
Writes <OUT>/survey100/vet/alias100.json. Usage: python -I survey100/alias100.py KEY [KEY ...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "followup"))
sys.path.insert(0, str(HERE))
import fu
from alias import joint_at
from common100 import VET

os.environ.setdefault("VET_LC_DIR", str(fu.DATA / "lc_cache" / "lc"))


def run(r):
    from get_lightcurve import get_lightcurve

    s = r["stats"]
    e = fu.fit_ephem(s)
    D, depth = s["dur_h"] / 24, s["depth"]
    # inject at the RAW depth (survey depth / survival s_det from rereduce.py) when it is known;
    # otherwise at the detected depth, which under-states what a real dip would give
    rj = VET / "rereduce_joint.json"
    raw = json.loads(rj.read_text()).get(r["key"], {}).get("d_raw") if rj.exists() else None
    inj_depth, inj_kind = (raw, "raw (rereduce s_det)") if raw and np.isfinite(raw) else (depth, "detected")
    lc, _ = get_lightcurve(r["tic"], 2_000_000_000)
    pp = fu.prep(lc)
    t = (pp["idx"] + 0.5) * fu.BIN_D
    tlo, thi = t.min(), t.max()
    det = list(s["epochs"])
    met = fu.Meter(pp, D, exclude=det)
    out = {"key": r["key"], "P": e["P"], "T0": e["T0"], "P_over_13.70": e["P"] / 13.70,
           "P_over_27.40": e["P"] / 27.40, "inj_depth": inj_depth, "inj_kind": inj_kind, "aliases": []}
    for k in (2, 3, 4, 5):
        m = np.arange(int(np.floor((tlo - e["T0"]) / e["P"])) - 1, int(np.ceil((thi - e["T0"]) / e["P"])) + 2)
        tt = [e["T0"] + (mm + j / k) * e["P"] for mm in m for j in range(1, k)]
        others = [float(x) for x in tt if met.at(float(x)) is not None]
        row = {"k": k, "P": e["P"] / k, "n_other": len(others)}
        if others:
            obs = joint_at(met, others)
            lci = lc
            for x in others:
                lci = fu.inject_box(lci, x, D, inj_depth)
            inj = joint_at(fu.Meter(fu.prep(lci), D, exclude=det + others), others)
            row.update(obs=obs, inj=inj)
            rec = inj is not None and inj["ses"] >= 4
            row["outcome"] = ("excluded" if rec and obs["ses"] < inj["ses"] - 3 else
                              "allowed (dip seen at the other epochs?)" if obs["ses"] >= 3 else
                              "allowed (not excluded)")
        else:
            row["outcome"] = "allowed (no other epoch in data)"
        out["aliases"].append(row)
    return out


def main():
    g = {json.loads(x)["key"]: json.loads(x) for x in open(VET / "graded.jsonl") if x.strip()}
    fp = VET / "alias100.json"
    res = json.loads(fp.read_text()) if fp.exists() else {}
    for k in sys.argv[1:]:
        try:
            res[k] = run(g[k])
        except Exception as ex:  # noqa: BLE001
            res[k] = {"key": k, "error": repr(ex)[:200]}
        fp.write_text(json.dumps(res, indent=1, default=float))
        a = res[k]
        print(k, "P/13.70 %.3f P/27.40 %.3f" % (a.get("P_over_13.70", np.nan), a.get("P_over_27.40", np.nan)),
              a.get("error", ""), flush=True)
        for row in a.get("aliases", []):
            o, i = row.get("obs") or {}, row.get("inj") or {}
            print(f"  P/{row['k']} = {row['P']:.3f} d: {row['outcome']}; n {row['n_other']}, obs SES "
                  f"{o.get('ses', np.nan):.1f}, injected SES {i.get('ses', np.nan):.1f}", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
