"""Evaluation harness (docs/ML_DESIGN.md §6). Reads score files only.

Inputs (§9): ``outputs/scores/<scorer>/sNNNN/{real,inj_<name>}.parquet`` with ``tic, sector,
inj_id, score`` (higher = weirder), the injection truth tables
``<data>/injections/<name>/sNNNN/{val,test}/injections.parquet`` and the sector metadata (for
Tmag / camera / CCD strata). No light curve is read, so every scorer is judged identically.

Ranking rule: an injected light curve's percentile is its mid-rank among the **real** light
curves of the **same split** and the **same stratum** (Tmag bin x camera/CCD,
``oddball.common.strata``): val injections against real val stars, test against test. It is
"in the top 1 %" when that percentile is >= 0.99. (Strata hold ~40-1000 real stars per split,
so the cut is coarse in the smallest strata and never looser than 1 %.)

AUROC per kind is the mean of those percentiles: the probability that the injection outranks a
random real star of its own split and stratum (ties count one half).

Known positives (``known`` split) are ranked against all real non-known stars of the sector in
their stratum. They are never used for tuning; this is the blind check.

Systematics veto (``veto``): flagged light curves, real and injected, drop to the bottom and
everything is re-ranked (``pct_veto``, ``top1%_veto``). ``cmv2`` (default) is the per-event
common-mode veto ``outputs/veto/cmv2/sNNNN/{real,inj_<name>}.parquet`` (``cmv2_flag``, joined by
TIC / ``inj_id``, so it is identical for every scorer); ``cmv3`` is its extension with
background, PDC/SAP, gap-edge, QUALITY, dump-edge, asteroid-track and same-column labels
(``outputs/veto/cmv3``, ML_DESIGN §13); ``sys_flag`` is the v0 per-star flag carried in the score
files; ``none`` disables it.

Injected SNR (PLAN gate clarification 2): ``snr_inj = depth / rms_bin(host) * sqrt(duration /
bin)``, the host's robust scatter of its binned, normalised PDCSAP flux from the sector metadata.
Diagnostic only: the gate is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from oddball.common.paths import data_root, outputs_root, sector_dir
from oddball.common.splits import split_array
from oddball.common.strata import TMAG_LABELS, stratum_id, tmag_bin
from oddball.tess.arrays import dir_metadata

# Half-decade depth bins centred on 0.3 %, 1 %, 3 %, 10 % (edges at geometric midpoints).
DEPTH_EDGES = (0.001, 0.00173, 0.00548, 0.0173, 0.0548, 0.173, 0.3001)
DEPTH_LABELS = ("0.1-0.17%", "~0.3%", "~1%", "~3%", "~10%", "17-30%")
DUR_EDGES = (2 / 24, 6 / 24, 1.0, 2.5, 5.0001)
DUR_LABELS = ("2-6 h", "6 h-1 d", "1-2.5 d", "2.5-5 d")
TOPS = {"top1%": 0.99, "top0.1%": 0.999}
GATE_DEPTH = "~0.3%"
GATE_RECALL = 0.5
SNR_EDGES = (0.0, 3.0, 10.0, 30.0, 100.0, np.inf)
SNR_LABELS = ("<3", "3-10", "10-30", "30-100", ">100")
VETOES = ("cmv2", "cmv3", "sys_flag", "none")
PER_EVENT_VETOES = ("cmv2", "cmv3")


@dataclass
class EvalData:
    scorer: str
    sector: int
    inj_name: str
    real: pd.DataFrame  # tic, score, split, stratum, tmag, ...
    inj: pd.DataFrame  # injection truth + score + pct
    veto: str = "cmv2"


def score_dir(scorer: str, sector: int) -> Path:
    return outputs_root() / "scores" / scorer / f"s{sector:04d}"


def load_truth(inj_name: str, sector: int) -> pd.DataFrame:
    parts = []
    for split in ("val", "test"):
        p = data_root() / "injections" / inj_name / f"s{sector:04d}" / split / "injections.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise FileNotFoundError(f"no injection truth for {inj_name} s{sector:04d}")
    return pd.concat(parts, ignore_index=True)


def midrank_pct(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    lo = np.searchsorted(ref_sorted, x, side="left")
    hi = np.searchsorted(ref_sorted, x, side="right")
    return (lo + hi) / 2.0 / max(len(ref_sorted), 1)


def rank_within(ref: pd.DataFrame, q: pd.DataFrame, keys: list[str]) -> np.ndarray:
    """Percentile of each ``q.score`` among ``ref.score`` rows sharing ``keys``."""
    out = np.full(len(q), np.nan)

    def norm(k):  # pandas returns 1-tuples for some single-key groupbys and scalars for others
        return k if isinstance(k, tuple) else (k,)

    groups = {norm(k): np.sort(g.score.to_numpy()) for k, g in ref.groupby(keys)}
    for k, idx in q.groupby(keys).indices.items():
        r = groups.get(norm(k))
        if r is None or len(r) == 0:
            continue
        out[idx] = midrank_pct(r, q.score.to_numpy()[idx])
    return out


def _clean(s: pd.Series) -> pd.Series:
    # NaN scores rank lowest (a scorer that failed on a light curve did not find it)
    return s.astype(np.float64).fillna(-np.inf)


def veto_dir(name: str, sector: int) -> Path:
    return outputs_root() / "veto" / name / f"s{sector:04d}"


def load_veto(name: str, sector: int, inj_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(real: tic -> veto_flag, inj: inj_id -> veto_flag) for a veto table such as ``cmv2``."""
    d = veto_dir(name, sector)
    col = f"{name}_flag"
    r = pd.read_parquet(d / "real.parquet", columns=["tic", col])
    i = pd.read_parquet(d / f"inj_{inj_name}.parquet", columns=["inj_id", col])
    return (r.rename(columns={col: "veto_flag"}), i.rename(columns={col: "veto_flag"}))


def load(scorer: str, sector: int, inj_name: str, veto: str = "cmv2") -> EvalData:
    if veto not in VETOES:
        raise ValueError(f"veto must be one of {VETOES}")
    sd = score_dir(scorer, sector)
    real = pd.read_parquet(sd / "real.parquet")
    inj_scores = pd.read_parquet(sd / f"inj_{inj_name}.parquet")
    meta_full = dir_metadata(sector_dir(sector))
    meta = meta_full[["tic", "tmag", "camera", "ccd"]]

    real = real.drop(columns=[c for c in ("tmag", "camera", "ccd", "stratum") if c in real])
    real = real.merge(meta, on="tic", how="left", validate="one_to_one")
    real["score"] = _clean(real.score)
    real["split"] = split_array(real.tic.to_numpy())
    real["stratum"] = stratum_id(real.tmag, real.camera, real.ccd)

    truth = load_truth(inj_name, sector)
    keep = ["inj_id", "score"] + [c for c in inj_scores.columns if c in ("sys_flag", "sys_frac",
                                                                        "cm_frac", "dump_frac")]
    inj = truth.merge(inj_scores[keep], on="inj_id", how="left", validate="one_to_one")
    n_missing = int(inj.score.isna().sum())
    if n_missing:
        print(f"warning: {n_missing} injections have no score (ranked lowest)")
    inj["score"] = _clean(inj.score)
    inj["stratum"] = stratum_id(inj.tmag, inj.camera, inj.ccd)
    ref = real[real.split.isin(["val", "test"])]
    inj["pct"] = rank_within(ref, inj, ["split", "stratum"])
    inj["depth_bin"] = pd.cut(inj.depth, DEPTH_EDGES, labels=DEPTH_LABELS, right=False)
    inj["dur_bin"] = pd.cut(inj.duration_d, DUR_EDGES, labels=DUR_LABELS, right=False)
    inj["tmag_bin"] = np.asarray(TMAG_LABELS, dtype=object)[tmag_bin(inj.tmag)]
    for name, cut in TOPS.items():
        inj[name] = inj.pct >= cut
    host = meta_full[["tic", "rms_bin", "bin_minutes"]].rename(columns={"tic": "host_tic"})
    inj = inj.merge(host, on="host_tic", how="left", validate="many_to_one")
    with np.errstate(invalid="ignore", divide="ignore"):
        inj["snr_inj"] = (inj.depth / inj.rms_bin
                          * np.sqrt(inj.duration_d / (inj.bin_minutes / 1440.0)))  # fmt: skip
    inj["snr_bin"] = pd.cut(inj.snr_inj, SNR_EDGES, labels=SNR_LABELS, right=False)

    if veto in PER_EVENT_VETOES:
        vr, vi = load_veto(veto, sector, inj_name)
        real = real.merge(vr, on="tic", how="left", validate="one_to_one")
        inj = inj.merge(vi, on="inj_id", how="left", validate="one_to_one")
        n_miss = int(inj.veto_flag.isna().sum()) + int(real.veto_flag.isna().sum())
        if n_miss:
            print(f"warning: {n_miss} light curves have no {veto} entry (not vetoed)")
        real["veto_flag"] = real.veto_flag.fillna(False).astype(bool)
        inj["veto_flag"] = inj.veto_flag.fillna(False).astype(bool)
    elif veto == "sys_flag" and "sys_flag" in inj and "sys_flag" in real:
        real["veto_flag"] = real.sys_flag.astype(bool)
        inj["veto_flag"] = inj.sys_flag.astype(bool)
    if "veto_flag" in inj:
        # Symmetric veto: flagged light curves (real and injected) drop to the bottom, then
        # everything is re-ranked, so the top-1% cut is recomputed without flagged real stars.
        ref = real[real.split.isin(["val", "test"])]
        rv = ref.assign(score=np.where(ref.veto_flag, -np.inf, ref.score))
        iv = inj.assign(score=np.where(inj.veto_flag, -np.inf, inj.score))
        inj["pct_veto"] = rank_within(rv, iv, ["split", "stratum"])
        inj["top1%_veto"] = (inj.pct_veto >= TOPS["top1%"]) & ~inj.veto_flag
    else:
        veto = "none"
    return EvalData(scorer, sector, inj_name, real, inj, veto)


def recall_table(inj: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = inj.groupby(by, observed=True)
    out = g.agg(n=("inj_id", "size"), **{k: (k, "mean") for k in TOPS}, auroc=("pct", "mean"))
    return out.reset_index()


def known_table(real: pd.DataFrame, sector: int) -> pd.DataFrame:
    kp = pd.read_csv(Path(__file__).resolve().parents[3] / "docs" / "known_positives.csv")
    known = real[real.split == "known"].copy()
    ref = real[real.split != "known"]
    known["pct_stratum"] = rank_within(ref, known, ["stratum"])
    known["pct_global"] = midrank_pct(np.sort(ref.score.to_numpy()), known.score.to_numpy())
    known = known.merge(kp[["tic", "name", "class"]], on="tic", how="left")
    known["top1%"] = known.pct_stratum >= 0.99
    cols = ["name", "tic", "class", "tmag", "camera", "ccd", "score", "pct_stratum",
            "pct_global", "top1%"]  # fmt: skip
    return known[cols].sort_values("pct_stratum", ascending=False).reset_index(drop=True)


def gate(ed: EvalData, known: pd.DataFrame, split: str = "test") -> dict:
    astro = ed.inj[(~ed.inj.is_null) & (ed.inj.split == split)]
    at = astro[astro.depth_bin == GATE_DEPTH]
    rec = float(at["top1%"].mean()) if len(at) else float("nan")
    per_kind = at.groupby("kind")["top1%"].mean().to_dict()
    frac_known = float(known["top1%"].mean()) if len(known) else float("nan")
    return {
        "split": split,
        "inj_recall_top1_at_0.3pct": rec,
        "n_at_0.3pct": len(at),
        "per_kind": per_kind,
        "inj_pass": bool(rec >= GATE_RECALL),
        "known_in_top1": int(known["top1%"].sum()),
        "n_known": len(known),
        "known_frac_top1": frac_known,
        "known_pass_majority": bool(frac_known >= 0.5),
        "known_pass_all": bool(len(known) and known["top1%"].all()),
        "pass": bool(rec >= GATE_RECALL and frac_known >= 0.5),
    }


def null_table(inj: pd.DataFrame) -> pd.DataFrame:
    nul = inj[inj.is_null]
    if nul.empty:
        return pd.DataFrame()
    g = nul.groupby(["split", "kind", "event_id"])
    out = g.agg(n=("inj_id", "size"), depth=("depth", "first"), duration_d=("duration_d", "first"),
                t0_btjd=("t0_btjd", "first"), camera=("camera", "first"), ccd=("ccd", "first"),
                **{k: (k, "mean") for k in TOPS}, median_pct=("pct", "median"))  # fmt: skip
    if "sys_flag" in nul:
        out["sys_flag_rate"] = g.sys_flag.mean()
    if "veto_flag" in nul:
        out["veto_flag_rate"] = g.veto_flag.mean()
    return out.reset_index()


def known_veto(ed: EvalData) -> pd.DataFrame:
    """Known positives: veto flag and percentile after the veto (blind check, never tuned on)."""
    if "veto_flag" not in ed.real:
        return pd.DataFrame()
    real = ed.real.assign(score=np.where(ed.real.veto_flag, -np.inf, ed.real.score))
    known = real[real.split == "known"].copy()
    ref = real[real.split != "known"]
    known["pct_veto"] = rank_within(ref, known, ["stratum"])
    kp = pd.read_csv(Path(__file__).resolve().parents[3] / "docs" / "known_positives.csv")
    known = known.merge(kp[["tic", "name"]], on="tic", how="left")
    known["top1%_veto"] = (known.pct_veto >= 0.99) & ~known.veto_flag
    return known[["name", "tic", "veto_flag", "pct_veto", "top1%_veto"]]


def snr_table(inj: pd.DataFrame, split: str = "test", depth_bin: str | None = None,
              col: str = "top1%") -> pd.DataFrame:
    """Recall vs injected SNR (non-null injections of one split, optionally one depth bin)."""
    a = inj[(~inj.is_null) & (inj.split == split)]
    if depth_bin is not None:
        a = a[a.depth_bin == depth_bin]
    g = a.groupby("snr_bin", observed=False)
    return pd.DataFrame({"n": g.size(), "recall": g[col].mean()}).reindex(list(SNR_LABELS))
