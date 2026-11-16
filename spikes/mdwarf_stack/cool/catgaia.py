"""Gaia DR3 tables through VizieR (CfA / IDIA mirrors), for when the ESA Gaia TAP server cannot be reached
(16 Nov 2026: gea.esac.esa.int and tapvizier.cds.unistra.fr timed out from here; the US/SA mirrors answered).

Cone queries around each star's Gaia DR3 position (gaia_match.csv), then rows are matched to the target by
source_id (DR3 positions are epoch 2016, the same as the cone centres).
  gaiadr3_v  I/355/gaiadr3  25"  target row -> gaia_tap_gaia_source.csv; all rows kept for neighbour astrometry
  veb_v      I/358/veb       3"  -> gaia_tap_veb.csv          (Gaia DR3 eclipsing binaries)
  vclass_v   I/358/vclassre  3"  -> gaia_tap_vclass.csv       (variability classifier)
  nss_v      I/357/tbooc     3"  -> gaia_tap_nss.csv          (two-body orbits)
  vrot_v     I/358/vrot      3"  -> gaia_tap_vrot.csv         (rotational modulation)
Column names are mapped to the ESA names catsum.py reads. status.json gets gaia_tap_<name> with via=VizieR.

Usage: python -I cool/catgaia.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catpass as C

C.R.CATALOGUES.update({
    "gaiadr3_v": ("I/355/gaiadr3", 25.0, ["**"], {}),
    "veb_v": ("I/358/veb", 3.0, ["**"], {}),
    "vclass_v": ("I/358/vclassre", 3.0, ["**"], {}),
    "nss_v": ("I/357/tbooc", 3.0, ["**"], {}),
    "vrot_v": ("I/358/vrot", 3.0, ["**"], {}),
})  # fmt: skip

# VizieR column -> ESA column (first match wins)
MAP = {
    "gaia_source": {"Source": "source_id", "RUWE": "ruwe", "RV": "radial_velocity", "e_RV": "radial_velocity_error",
                    "o_RV": "rv_nb_transits", "RVamp": "rv_amplitude_robust", "RVchi2": "rv_chisq_pvalue",
                    "IPDfmp": "ipd_frac_multi_peak", "NSS": "non_single_star", "VarFlag": "phot_variable_flag",
                    "BP-RP": "bp_rp", "Gmag": "phot_g_mean_mag", "Plx": "parallax", "e_Plx": "parallax_error",
                    "pmRA": "pmra", "pmDE": "pmdec", "RA_ICRS": "ra", "DE_ICRS": "dec"},
    "veb": {"Source": "source_id", "Freq": "frequency", "Per": "period"},
    "vclass": {"Source": "source_id", "Class": "best_class_name", "ClassSc": "best_class_score",
               "Classifier": "classifier_name"},
    "nss": {"Source": "source_id", "NSSmodel": "nss_solution_type", "Per": "period", "e_Per": "period_error",
            "ecc": "eccentricity", "K1": "semi_amplitude_primary", "K2": "semi_amplitude_secondary",
            "Sig": "significance"},
    "vrot": {"Source": "source_id", "Prot": "best_rotation_period", "PRot": "best_rotation_period",
             "MaxAI": "max_activity_index"},
}  # fmt: skip
KEYS = {"gaia_source": "gaiadr3_v", "veb": "veb_v", "vclass": "vclass_v", "nss": "nss_v", "vrot": "vrot_v"}


def main():
    stars = C.star_list()
    sg = C.gpos(stars)
    m = pd.read_csv(C.CATD / "gaia_match.csv", dtype={"source_id": "Int64"})
    sid = dict(zip(m.tic, m.source_id, strict=True))
    for name, key in KEYS.items():
        try:
            df = C.R.query_catalogue(key, sg, chunk=40, sleep=1.0)
            src_col = "Source" if "Source" in df.columns else next(c for c in df.columns if c.lower() == "source")
            df[src_col] = pd.to_numeric(df[src_col], errors="coerce").astype("Int64")
            if name == "gaia_source":
                df.to_csv(C.CATD / "gaiadr3_cone25.csv", index=False)
            on = df[[sid.get(t) is not pd.NA and s == sid.get(t) for t, s in zip(df.tic, df[src_col], strict=True)]]
            ren = {}
            for vz, esa in MAP[name].items():
                if vz in on.columns and esa not in ren.values():
                    ren[vz] = esa
            out = on.rename(columns=ren)
            if name == "veb" and "frequency" not in out.columns and "period" in out.columns:
                out["frequency"] = 1 / pd.to_numeric(out.period, errors="coerce")
            if name == "vclass" and "classifier_name" not in out.columns:
                out["classifier_name"] = ""
            out = out.drop_duplicates("source_id")
            out.to_csv(C.CATD / f"gaia_tap_{name}.csv", index=False)
            missing = sorted(set(MAP[name].values()) - set(out.columns))
            C.status_update(f"gaia_tap_{name}", {"ok": True, "rows": len(out), "via": f"VizieR {C.R.CATALOGUES[key][0]}",
                                                 "cone_rows": len(df), "missing_cols": missing})  # fmt: skip
            print(name, "ok", len(out), "of cone rows", len(df), "missing", missing, "cols", list(df.columns)[:40])
        except Exception as e:  # noqa: BLE001
            C.status_update(f"gaia_tap_{name}", {"ok": False, "error": repr(e)[:300], "via": "VizieR"})
            print(name, "ERROR", repr(e)[:300])
    _ = np  # keep import for column math in future edits


if __name__ == "__main__":
    main()
