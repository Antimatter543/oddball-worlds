"""Write ``cmv2_flag`` / ``cmv3_flag`` for a sector with the rule chosen on S32 val (no tuning).

    uv run python -I scripts/apply_veto_rule.py --sector 43            # both vetoes
    uv run python -I scripts/apply_veto_rule.py --sector 43 --veto cmv3

``tune_veto.py`` / ``tune_veto3.py`` choose the star flag on a sector's val injections and write
it. Other sectors have no injection sets, and the search must not re-tune on them: this script
reads the frozen rule from the reference sector's provenance (``outputs/veto/<veto>/s0032/
provenance.json``, key ``flag``) and applies it unchanged to ``outputs/veto/<veto>/sNNNN/
real.parquet`` (built by ``build_veto.py`` / ``build_veto3.py``). For ``cmv3`` the chosen
configuration's columns are copied to ``cmv3_*`` as ``tune_veto3.py`` does.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from oddball.common.paths import outputs_root
from oddball.score import veto as V
from oddball.score import veto3 as V3

COPY3 = ("res_snr", "sys_snr", "res_snr_pdc", "sys_snr_pdc", "res_t", "res_ev", "sys_t", "sys_ev")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def frozen_rule(veto: str, ref_sector: int) -> dict:
    p = outputs_root() / "veto" / veto / f"s{ref_sector:04d}" / "provenance.json"
    flag = json.loads(p.read_text()).get("flag")
    if not flag:
        raise SystemExit(f"{p} has no tuned flag rule")
    keep = ("rule", "config", "labels", "t_sys", "t_res", "pdc_only", "tuned_utc")
    return {k: flag[k] for k in keep if k in flag} | {"source": str(p)}


def apply(veto: str, sector: int, ref_sector: int) -> dict:
    d = outputs_root() / "veto" / veto / f"s{sector:04d}"
    rp = d / "real.parquet"
    if not rp.exists():
        raise SystemExit(f"no {rp}: run build_veto{'3' if veto == 'cmv3' else ''}.py first")
    rule = frozen_rule(veto, ref_sector)
    df = pd.read_parquet(rp)
    if veto == "cmv2":
        df["cmv2_flag"] = V.veto_flag(df, rule["t_sys"], rule["t_res"], rule["pdc_only"])
    else:
        cfg = rule["config"]
        if f"{cfg}_sys_snr" not in df:
            raise SystemExit(f"{rp} lacks configuration {cfg!r} (build_veto3.py --configs)")
        df["cmv3_flag"] = V3.veto_flag(df, cfg, rule["t_sys"], rule["t_res"], rule["pdc_only"])
        for c in COPY3:
            df[f"cmv3_{c}"] = df[f"{cfg}_{c}"]
    df.to_parquet(rp, index=False)
    frac = float(df[f"{veto}_flag"].mean())
    pp = d / "provenance.json"
    prov = json.loads(pp.read_text()) if pp.exists() else {}
    prov["flag"] = {**rule, "fixed_from_sector": ref_sector, "tuned_here": False,
                    "flag_frac_real_all": frac, "n_real": len(df),
                    "applied_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds")}  # fmt: skip
    pp.write_text(json.dumps(prov, indent=1, default=float))
    log(f"{veto} s{sector:04d}: rule from s{ref_sector:04d} {rule.get('config', '')} t_sys "
        f"{rule['t_sys']} t_res {rule['t_res']} pdc_only {rule['pdc_only']}: flags {frac:.4f} "
        f"of {len(df)} real light curves")
    return prov["flag"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--veto", nargs="+", default=["cmv2", "cmv3"], choices=["cmv2", "cmv3"])
    ap.add_argument("--ref-sector", type=int, default=32)
    a = ap.parse_args()
    if a.sector == a.ref_sector:
        raise SystemExit("the reference sector's flag is written by tune_veto*.py")
    for v in a.veto:
        apply(v, a.sector, a.ref_sector)


if __name__ == "__main__":
    main()
