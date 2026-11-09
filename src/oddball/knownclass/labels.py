"""Catalogue types -> known classes, and the hard exclusion list (docs/ML_DESIGN.md §4.5).

Known classes are the common, *boring* variable types a score should be able to explain away.
The anomaly targets (dippers and other young-star dimmers, Boyajian-like stars, exocomets,
disintegrating planets) must stay out-of-distribution: a star that any source puts in one of
those classes is **excluded**, i.e. it is neither a training label nor a "quiet" star.

Per TIC, label priority is Prsa+2022 TESS EB > VSX (nearest entry within ``VSX_LABEL_RADIUS``)
> Gaia DR3 SOS tables (``vari_summary.in_vari_*``) > Gaia DR3 classifier best class (score >=
``GAIA_MIN_SCORE``). Exclusions use wider nets: any VSX entry within ``VSX_RADIUS`` (one TESS
pixel) or any Gaia source within ``GAIA_RADIUS`` with an excluded type excludes the TIC, since
its variability would be blended into our light curve.

``quiet`` and ``other`` are assigned per light curve at training time (``model.training_targets``).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------
# Classes

KNOWN_CLASSES = ("EB", "DSCT_GDOR", "RR", "CEP", "LPV", "ROT", "ELL", "BCEP_SPB", "quiet")
# "other": unlabelled, non-quiet light curves. A catch-all so that ordinary stars (and stars with
# events no class explains) are not forced into a known class. It never explains anything away.
OTHER = "other"
CLASSES = (*KNOWN_CLASSES, OTHER)

VSX_CLASS = {
    "EB": {"E", "EA", "EB", "EW", "ED", "ESD", "EAD", "EC", "EL"},
    "DSCT_GDOR": {"DSCT", "DSCTC", "DSCTR", "HADS", "GDOR", "SXPHE", "ROAP", "ROAM"},
    "RR": {"RR", "RRAB", "RRC", "RRD", "RRA", "RRB", "RRAB/BL"},
    "CEP": {"CEP", "DCEP", "DCEPS", "DCEPS(B)", "DCEP(B)", "CW", "CWA", "CWB", "ACEP", "RV",
            "RVA", "RVB", "T2CEP", "BLHER", "BLAP"},
    "LPV": {"M", "SR", "SRA", "SRB", "SRC", "SRD", "SRS", "L", "LB", "LC", "LPV"},
    "ROT": {"ROT", "BY", "RS", "ACV", "ACVO", "SXARI", "CP", "FKCOM", "MCP"},
    "ELL": {"ELL", "R"},
    "BCEP_SPB": {"BCEP", "BCEPS", "SPB", "SPBE", "LPB"},
}  # fmt: skip

GAIA_CLASSIFIER = {
    "ECL": "EB",
    "DSCT|GDOR|SXPHE": "DSCT_GDOR",
    "RR": "RR",
    "CEP": "CEP",
    "LPV": "LPV",
    "SOLAR_LIKE": "ROT",
    "RS": "ROT",
    "ACV|CP|MCP|ROAM|ROAP|SXARI": "ROT",
    "ELL": "ELL",
    "SPB": "BCEP_SPB",
    "BCEP": "BCEP_SPB",
}
GAIA_SOS = {  # vari_summary flag -> class (checked in this order)
    "in_vari_eclipsing_binary": "EB",
    "in_vari_cepheid": "CEP",
    "in_vari_rrlyrae": "RR",
    "in_vari_long_period_variable": "LPV",
    "in_vari_ms_oscillator": "DSCT_GDOR",
    "in_vari_rotation_modulation": "ROT",
}
GAIA_MIN_SCORE = 0.5
VSX_LABEL_RADIUS = 5.0  # arcsec
GAIA_LABEL_RADIUS = 2.0  # arcsec
GAIA_G_MINUS_T = (-0.5, 2.5)  # plausible G - Tmag for the target's own Gaia counterpart

# ---------------------------------------------------------------------------------------------
# Hard exclusions: anomaly-target-like and irregular-dimming types (never a label, never quiet)

# VSX type tokens (after splitting combined types on | + / , and dropping ':' uncertainty marks)
VSX_EXCLUDE_EXACT = {
    "YSO", "TTS", "CTTS", "WTTS", "CTTS/ROT", "TTS/ROT", "DYPER", "FUOR", "FU", "EXOR", "UXOR",
    "UX", "ORION", "INS", "INSA", "INSB", "INST", "INT", "IN", "INA", "INB", "INSA(YY)",
    "I", "IA", "IB", "IS", "ISA", "ISB", "DIP", "DIPPER", "RCB", "BOY", "ZZO",
    "HAEBE", "HAE", "HBE", "GTTS",
}  # fmt: skip
VSX_EXCLUDE_PREFIX = ("IN", "YSO", "TTS", "CTTS", "WTTS", "FU", "EXOR", "UXOR", "DIP")
GAIA_EXCLUDE = {"YSO", "RCB"}

_SPLIT = re.compile(r"[|+,/ ]+")


def vsx_tokens(type_str: str) -> list[tuple[str, bool]]:
    """Split a VSX type into ``(token, uncertain)`` pairs (``EA:|ROT`` -> EA uncertain, ROT)."""
    s = str(type_str or "").strip().upper()
    if not s or s in ("NAN", "NONE"):
        return []
    whole_unc = s.endswith(":")
    out = []
    for tok in _SPLIT.split(s):
        if not tok:
            continue
        unc = whole_unc or tok.endswith(":")
        tok = tok.rstrip(":")
        if tok:
            out.append((tok, unc))
    return out


def vsx_excluded(type_str: str) -> bool:
    """True if any token (certain or not) is an excluded young-star / irregular-dimming type."""
    s = str(type_str or "").upper()
    for tok, _ in vsx_tokens(s):
        if tok in VSX_EXCLUDE_EXACT or tok.startswith(VSX_EXCLUDE_PREFIX):
            return True
    # GCVS-style combined types like "CTTS/ROT" are caught token-wise above; also catch the
    # whole string for safety.
    return s.rstrip(":") in VSX_EXCLUDE_EXACT


def vsx_class(type_str: str) -> str | None:
    """Known class of a VSX type, or None (unused or uncertain). Eclipses take precedence."""
    toks = vsx_tokens(type_str)
    if not toks or vsx_excluded(type_str):
        return None
    if any(unc for _, unc in toks):
        return None
    full = str(type_str).strip().upper()
    for cls, names in VSX_CLASS.items():  # exact full-string match first (e.g. "RRAB/BL")
        if full in names:
            return cls
    found = [cls for tok, _ in toks for cls, names in VSX_CLASS.items() if tok in names]
    if not found:
        return None
    return "EB" if "EB" in found else found[0]


def gaia_class(row) -> str | None:
    for flag, cls in GAIA_SOS.items():
        if _truthy(row.get(flag)):
            return cls
    name = row.get("best_class_name")
    score = row.get("best_class_score")
    if isinstance(name, str) and name in GAIA_CLASSIFIER and _num(score) >= GAIA_MIN_SCORE:
        return GAIA_CLASSIFIER[name]
    return None


def _truthy(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, str):
        return v.strip().lower() in ("true", "t", "1")
    return bool(v)


def _num(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


# ---------------------------------------------------------------------------------------------
# Per-TIC label table


def build_labels(tics: np.ndarray, tmag: np.ndarray, vsx: pd.DataFrame | None,
                 prsa: pd.DataFrame | None, gaia: pd.DataFrame | None,
                 always_exclude: dict[int, str]) -> pd.DataFrame:  # fmt: skip
    """One row per TIC: catalogue label (or empty), source, exclusion flag and reason.

    ``always_exclude``: TIC -> reason (known positives, injection hosts).
    """
    df = pd.DataFrame({"tic": np.asarray(tics, np.int64), "tmag": np.asarray(tmag, float)})
    df["label"] = ""
    df["label_source"] = ""
    df["label_detail"] = ""
    df["excluded"] = False
    df["exclude_reason"] = ""
    df["flagged_any"] = False  # any catalogue says "variable" (so the star can't be quiet)
    df["n_sources_disagree"] = 0
    idx = pd.Series(np.arange(len(df)), index=df.tic)
    cand = {}  # tic -> list of (priority, cls, source, detail)

    def add(tic, prio, cls, src, detail):
        cand.setdefault(int(tic), []).append((prio, cls, src, detail))

    def exclude(tic_arr, reason):
        i = idx.reindex(np.asarray(tic_arr, np.int64)).dropna().astype(int).to_numpy()
        new = ~df.excluded.to_numpy()[i]
        df.loc[i[new], "exclude_reason"] = reason
        df.loc[i, "excluded"] = True

    ae = {int(k): v for k, v in always_exclude.items()}
    for reason in sorted(set(ae.values())):  # first, so these reasons win
        exclude([t for t, r in ae.items() if r == reason], reason)

    if prsa is not None and len(prsa):
        for r in prsa.itertuples():
            add(r.TIC, 0, "EB", "prsa2022", f"P={_num(r.Per):.4g} morph={_num(r.Morph):.2f}")
        df.loc[df.tic.isin(prsa.TIC.astype(np.int64)), "flagged_any"] = True

    if vsx is not None and len(vsx) and "Type" in vsx:
        v = vsx.copy()
        v["excl"] = v.Type.map(vsx_excluded)
        exclude(v.tic[v.excl], "vsx_type")
        df.loc[df.tic.isin(v.tic), "flagged_any"] = True
        near = v[v.sep_arcsec <= VSX_LABEL_RADIUS].sort_values("sep_arcsec")
        near = near.drop_duplicates("tic", keep="first")  # nearest entry only
        for r in near.itertuples():
            cls = vsx_class(r.Type)
            if cls:
                add(r.tic, 1, cls, "vsx", f"{r.Name} [{r.Type}]")

    if gaia is not None and len(gaia):
        g = gaia.copy()
        bad = g.best_class_name.isin(GAIA_EXCLUDE)
        exclude(g.tic[bad], "gaia_class")
        gt = g.gmag - g.tmag  # NaN when the source carries no photometry (VizieR vclassre)
        ok = (g.sep_arcsec <= GAIA_LABEL_RADIUS) & (gt.between(*GAIA_G_MINUS_T) | gt.isna())
        cp = g[ok].sort_values("sep_arcsec").drop_duplicates("tic", keep="first")
        var = cp.varflag.astype(str).str.upper().eq("VARIABLE") | cp.best_class_name.notna()
        df.loc[df.tic.isin(cp.tic[var]), "flagged_any"] = True
        for r in cp.to_dict("records"):
            cls = gaia_class(r)
            if cls is None:
                continue
            sos = any(_truthy(r.get(f)) for f in GAIA_SOS)
            src = "gaia_sos" if sos else "gaia_classifier"
            add(r["tic"], 2 if sos else 3, cls, src,
                f"DR3 {r['source_id']} {r.get('best_class_name')} {_num(r.get('best_class_score')):.2f}")

    for tic, items in cand.items():
        if tic not in idx.index:
            continue
        items.sort(key=lambda x: x[0])
        _prio, cls, src, detail = items[0]
        i = idx[tic]
        df.at[i, "label"] = cls
        df.at[i, "label_source"] = src
        df.at[i, "label_detail"] = detail
        df.at[i, "n_sources_disagree"] = len({c for _, c, _, _ in items if c != cls})

    df["label_any"] = df.label  # before exclusion: diagnostics only (e.g. injection hosts)
    df.loc[df.excluded, "label"] = ""
    return df
