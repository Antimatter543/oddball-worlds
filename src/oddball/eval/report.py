"""Markdown reports and plots for the evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oddball.common.strata import TMAG_LABELS
from oddball.eval.harness import (
    DEPTH_LABELS,
    DUR_LABELS,
    GATE_DEPTH,
    EvalData,
    gate,
    known_table,
    null_table,
    recall_table,
)
from oddball.inject.sampling import KINDS

SCORECARD_DEPTHS = ("~0.3%", "~1%", "~3%")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def pivot(inj: pd.DataFrame, col: str, labels, metric: str, kinds=KINDS) -> pd.DataFrame:
    t = recall_table(inj, ["kind", col])
    p = t.pivot(index="kind", columns=col, values=metric).reindex(index=list(kinds),
                                                                  columns=list(labels))
    n = t.pivot(index="kind", columns=col, values="n").reindex(index=list(kinds),
                                                               columns=list(labels))
    pooled = recall_table(inj, [col]).set_index(col).reindex(list(labels))
    p.loc["ALL (non-null)"] = pooled[metric]
    n.loc["ALL (non-null)"] = pooled["n"]
    return p, n


def _pivot_md(p: pd.DataFrame, n: pd.DataFrame) -> str:
    df = p.copy()
    df.columns = [str(c) for c in df.columns]
    df.insert(0, "kind", df.index)
    out = md_table(df.reset_index(drop=True))
    ncol = n.loc["ALL (non-null)"]
    return out + "\n\n_n per bin (all kinds): " + ", ".join(
        f"{c}: {int(v) if np.isfinite(v) else 0}" for c, v in ncol.items()) + "_"  # fmt: skip


def scorecard(ed: EvalData, split: str = "test") -> pd.DataFrame:
    astro = ed.inj[(~ed.inj.is_null) & (ed.inj.split == split)]
    p, _ = pivot(astro, "depth_bin", DEPTH_LABELS, "top1%")
    return p[list(SCORECARD_DEPTHS)]


def null_fpr(ed: EvalData, split: str) -> dict:
    nul = ed.inj[ed.inj.is_null & (ed.inj.split == split)]
    out = {"all": float(nul["top1%"].mean()) if len(nul) else float("nan")}
    for k, g in nul.groupby("kind"):
        out[k] = float(g["top1%"].mean())
    return out


# --------------------------------------------------------------------------------------------
# Plots


def _plot_recall_curves(ax, inj, col, labels, group, groups, metric="top1%"):
    x = np.arange(len(labels))
    for gname in groups:
        sub = inj[inj[group] == gname]
        t = recall_table(sub, [col]).set_index(col).reindex(list(labels))
        ax.plot(x, t[metric], marker="o", label=str(gname))
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
    ax.grid(alpha=0.3)


def plots(ed: EvalData, known: pd.DataFrame, out: Path, split: str = "test") -> list[str]:
    files = []
    astro = ed.inj[(~ed.inj.is_null) & (ed.inj.split == split)]

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, metric in zip(axs, ("top1%", "top0.1%"), strict=True):
        _plot_recall_curves(ax, astro, "depth_bin", DEPTH_LABELS, "kind", KINDS, metric)
        ax.set_title(f"{ed.scorer}: recall in {metric} vs depth ({split})")
        ax.set_xlabel("injected depth")
    axs[0].set_ylabel("recall")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "recall_vs_depth.png", dpi=110)
    plt.close(fig)
    files.append("recall_vs_depth.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    _plot_recall_curves(axs[0], astro, "depth_bin", DEPTH_LABELS, "tmag_bin", TMAG_LABELS)
    axs[0].set_title(f"recall in top 1% vs depth, by Tmag ({split}, all non-null kinds)")
    axs[0].legend(title="Tmag", fontsize=8)
    _plot_recall_curves(axs[1], astro, "dur_bin", DUR_LABELS, "kind", KINDS)
    axs[1].set_title(f"recall in top 1% vs duration, by kind ({split})")
    axs[1].legend(fontsize=8)
    axs[0].set_ylabel("recall")
    fig.tight_layout()
    fig.savefig(out / "recall_by_tmag_duration.png", dpi=110)
    plt.close(fig)
    files.append("recall_by_tmag_duration.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    bins = np.linspace(0, 1, 41)
    nul = ed.inj[ed.inj.is_null & (ed.inj.split == split)]
    axs[0].hist(astro.pct, bins=bins, density=True, alpha=0.5, label="astrophysical")
    for k, g in nul.groupby("kind"):
        axs[0].hist(g.pct, bins=bins, density=True, histtype="step", lw=2, label=k)
    axs[0].axvline(0.99, color="k", ls="--", lw=0.8)
    axs[0].set_xlabel("percentile vs real stars (same split, stratum)")
    axs[0].set_title("injection percentiles")
    axs[0].legend(fontsize=8)
    nt = null_table(ed.inj[ed.inj.split == split])
    if len(nt):
        lab = [f"e{r.event_id} {r.kind[5:]}\n{r.depth * 100:.2g}% {r.duration_d * 24:.0f}h"
               for r in nt.itertuples()]  # fmt: skip
        axs[1].bar(np.arange(len(nt)), nt["top1%"], color="tab:red")
        axs[1].set_xticks(np.arange(len(nt)), lab, fontsize=7)
        axs[1].set_ylim(0, 1)
        axs[1].set_ylabel("fraction in top 1%")
        axs[1].set_title("null false-positive rate per event")
    fig.tight_layout()
    fig.savefig(out / "null_events.png", dpi=110)
    plt.close(fig)
    files.append("null_events.png")

    if len(known):
        fig, ax = plt.subplots(figsize=(10, 0.3 * len(known) + 1.5))
        y = np.arange(len(known))
        ax.barh(y, known.pct_stratum, color=np.where(known["top1%"], "tab:green", "tab:gray"))
        ax.set_yticks(y, [f"{n} (T={t:.1f})" for n, t in zip(known.name, known.tmag,
                                                              strict=True)], fontsize=8)
        ax.axvline(0.99, color="k", ls="--", lw=0.8)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_xlabel("percentile within stratum (all real stars)")
        ax.set_title(f"{ed.scorer}: known positives in S{ed.sector}")
        fig.tight_layout()
        fig.savefig(out / "known_positives.png", dpi=110)
        plt.close(fig)
        files.append("known_positives.png")
    return files


# --------------------------------------------------------------------------------------------
# Single-scorer report


def write_report(ed: EvalData, out: Path, command: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    known = known_table(ed.real, ed.sector)
    gates = {s: gate(ed, known, s) for s in ("test", "val")}
    figs = plots(ed, known, out)

    full = recall_table(ed.inj, ["split", "is_null", "kind", "depth_bin", "tmag_bin"])
    full.to_csv(out / "recall_kind_depth_tmag.csv", index=False)
    recall_table(ed.inj, ["split", "is_null", "kind", "depth_bin"]).to_csv(
        out / "recall_kind_depth.csv", index=False)
    recall_table(ed.inj, ["split", "is_null", "kind", "dur_bin"]).to_csv(
        out / "recall_kind_duration.csv", index=False)
    known.to_csv(out / "known_positives.csv", index=False)
    nt = null_table(ed.inj)
    nt.to_csv(out / "null_events.csv", index=False)

    ref_sizes = ed.real[ed.real.split.isin(["val", "test"])].groupby(["split", "stratum"]).size()
    L = [f"# Evaluation: `{ed.scorer}` on S{ed.sector}, injection set `{ed.inj_name}`", ""]
    if command:
        L += [f"Command: `{command}`", ""]
    L += [
        (f"- Real light curves scored: {len(ed.real)} "
        f"(val {int((ed.real.split == 'val').sum())}, test {int((ed.real.split == 'test').sum())},"
        f" known {int((ed.real.split == 'known').sum())})"),
        f"- Injections: {len(ed.inj)} ("
        + ", ".join(f"{s}: {int((~g.is_null).sum())} astrophysical + {int(g.is_null.sum())} null"
                    for s, g in ed.inj.groupby("split")) + ")",
        ("- Rank rule: percentile among real stars of the same split and Tmag x CCD stratum; "
         "top 1% = percentile >= 0.99. Reference stratum sizes: ")
        + ", ".join(f"{s} min {int(g.min())} / median {int(g.median())}"
                    for s, g in ref_sizes.groupby(level=0)),
        "",
        "## Gate (PLAN.md)",
        "",
        ("Pass = recall in top 1% >= 50% for non-null injections in the ~0.3% depth bin "
         "(0.17-0.55%) **and** at least half of the S32 known positives in the top 1% of their "
         "stratum. (`known_pass_all` is the strict all-known reading.)"),
        "",
    ]  # fmt: skip
    g = pd.DataFrame([{k: v for k, v in gg.items() if k != "per_kind"} for gg in gates.values()])
    L += [md_table(g), ""]
    L += ["Per kind at ~0.3% (test): " + ", ".join(
        f"{k} {v:.2f}" for k, v in gates["test"]["per_kind"].items()), ""]  # fmt: skip

    for split in ("test", "val"):
        astro = ed.inj[(~ed.inj.is_null) & (ed.inj.split == split)]
        L += [f"## Recall by kind x depth ({split})", ""]
        for metric in ("top1%", "top0.1%"):
            p, n = pivot(astro, "depth_bin", DEPTH_LABELS, metric)
            L += [f"**{metric}**", "", _pivot_md(p, n), ""]
        if split == "val":
            break
        p, n = pivot(astro, "tmag_bin", TMAG_LABELS, "top1%")
        L += [f"## Recall in top 1% by kind x Tmag ({split}, all depths)", "", _pivot_md(p, n), ""]
        p, n = pivot(astro, "dur_bin", DUR_LABELS, "top1%")
        L += [f"## Recall in top 1% by kind x duration ({split}, all depths)", "",
              _pivot_md(p, n), ""]  # fmt: skip
        sub = astro[astro.depth_bin == GATE_DEPTH]
        p, n = pivot(sub, "tmag_bin", TMAG_LABELS, "top1%")
        L += [f"## Recall in top 1% by kind x Tmag at ~0.3% depth ({split})", "",
              _pivot_md(p, n), ""]  # fmt: skip

    L += ["## AUROC by kind (stratum-matched)", ""]
    au = recall_table(ed.inj, ["split", "kind"]).pivot(index="kind", columns="split",
                                                       values="auroc").reset_index()
    L += [md_table(au, "{:.3f}"), ""]

    L += ["## Null (instrument-like) injections", "",
          "False-positive rate = fraction of null injections in the top 1% (top 0.1%).", ""]
    for split in ("test", "val"):
        nul = ed.inj[ed.inj.is_null & (ed.inj.split == split)]
        rows = [{"split": split, "kind": k, "n": len(gg), "fpr_top1%": gg["top1%"].mean(),
                 "fpr_top0.1%": gg["top0.1%"].mean()} for k, gg in nul.groupby("kind")]
        rows.append({"split": split, "kind": "ALL", "n": len(nul),
                     "fpr_top1%": nul["top1%"].mean(), "fpr_top0.1%": nul["top0.1%"].mean()})
        L += [md_table(pd.DataFrame(rows)), ""]
    if len(nt):
        L += ["Per event:", "", md_table(nt, "{:.3g}"), ""]

    if "top1%_veto" in ed.inj:
        L += ["## Systematics flag (common-mode v0): with and without the veto", "",
              ("`veto` = light curves with `sys_flag` (real and injected) are sent to the bottom and "
               "everything is re-ranked; a flagged injection counts as not recovered. Shows what "
               "the flag buys on nulls and what it costs on astrophysical injections."), ""]
        rows = []
        for split in ("test", "val"):
            s = ed.inj[ed.inj.split == split]
            for lab, sel in (("null_ccd", s.kind == "null_ccd"),
                             ("null_dump", s.kind == "null_dump"),
                             ("astro (all depths)", ~s.is_null),
                             ("astro ~0.3%", (~s.is_null) & (s.depth_bin == GATE_DEPTH))):
                gg = s[sel]
                if gg.empty:
                    continue
                rows.append({"split": split, "set": lab, "n": len(gg),
                             "flag_rate": gg.sys_flag.mean(),
                             "top1%_without": gg["top1%"].mean(),
                             "top1%_with_veto": gg["top1%_veto"].mean()})
        L += [md_table(pd.DataFrame(rows), "{:.3f}"), ""]

    L += ["## Known positives in S32 (blind)", "", md_table(known, "{:.4g}"), ""]
    L += ["## Plots", ""] + [f"![{f}]({f})" for f in figs] + [""]
    (out / "report.md").write_text("\n".join(L))
    summary = {
        "scorer": ed.scorer, "sector": ed.sector, "inj": ed.inj_name, "gate": gates,
        "null_fpr": {s: null_fpr(ed, s) for s in ("test", "val")},
        "scorecard_test_top1": scorecard(ed, "test").round(3).to_dict(),
        "auroc_test": recall_table(ed.inj[ed.inj.split == "test"], ["kind"]).set_index(
            "kind").auroc.round(3).to_dict(),
        "known": known[["name", "pct_stratum", "pct_global"]].round(4).to_dict("records"),
    }  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


# --------------------------------------------------------------------------------------------
# Compare mode


def write_compare(eds: list[EvalData], out: Path, command: str = "", split: str = "test"):
    out.mkdir(parents=True, exist_ok=True)
    names = [e.scorer for e in eds]
    L = [f"# Scorer comparison on S{eds[0].sector} ({split}, injection set `{eds[0].inj_name}`)",
         ""]  # fmt: skip
    if command:
        L += [f"Command: `{command}`", ""]
    rows = []
    for e in eds:
        known = known_table(e.real, e.sector)
        g = gate(e, known, split)
        nf = null_fpr(e, split)
        astro = e.inj[(~e.inj.is_null) & (e.inj.split == split)]
        rows.append({
            "scorer": e.scorer, "recall@1% ~0.3%": g["inj_recall_top1_at_0.3pct"],
            "recall@1% all": astro["top1%"].mean(), "recall@0.1% all": astro["top0.1%"].mean(),
            "AUROC all": astro.pct.mean(), "null FPR": nf["all"],
            "null_ccd FPR": nf.get("null_ccd", np.nan), "null_dump FPR": nf.get("null_dump",
                                                                                np.nan),
            "known top1%": f"{g['known_in_top1']}/{g['n_known']}", "gate": g["pass"],
        })  # fmt: skip
        if "top1%_veto" in e.inj:
            nul = e.inj[e.inj.is_null & (e.inj.split == split)]
            at = astro[astro.depth_bin == GATE_DEPTH]
            rows[-1].update({"null FPR (veto)": nul["top1%_veto"].mean(),
                             "recall@1% all (veto)": astro["top1%_veto"].mean(),
                             "recall@1% ~0.3% (veto)": at["top1%_veto"].mean()})
    L += ["## Headline", "", md_table(pd.DataFrame(rows), "{:.3f}"), ""]

    for d in SCORECARD_DEPTHS:
        tab = pd.DataFrame({e.scorer: scorecard(e, split)[d] for e in eds})
        tab.insert(0, "kind", tab.index)
        L += [f"## Recall in top 1% at {d} depth", "", md_table(tab.reset_index(drop=True)), ""]
    tab = pd.DataFrame({e.scorer: recall_table(e.inj[e.inj.split == split], ["kind"])
                        .set_index("kind").auroc for e in eds})  # fmt: skip
    tab.insert(0, "kind", tab.index)
    L += ["## AUROC by kind", "", md_table(tab.reset_index(drop=True), "{:.3f}"), ""]
    kt = None
    for e in eds:
        k = known_table(e.real, e.sector)[["name", "pct_stratum"]].rename(
            columns={"pct_stratum": e.scorer})
        kt = k if kt is None else kt.merge(k, on="name")
    L += ["## Known positives (percentile within stratum)", "", md_table(kt, "{:.3f}"), ""]

    fig, axs = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    x = np.arange(len(DEPTH_LABELS))
    panels = [("ALL non-null", None), *[(k, k) for k in KINDS], ("null", "null")]
    for ax, (title, kind) in zip(axs.ravel(), panels, strict=True):
        for e in eds:
            s = e.inj[e.inj.split == split]
            if kind is None:
                s = s[~s.is_null]
            elif kind == "null":
                s = s[s.is_null]
            else:
                s = s[s.kind == kind]
            t = recall_table(s, ["depth_bin"]).set_index("depth_bin").reindex(list(DEPTH_LABELS))
            ax.plot(x, t["top1%"], marker="o", label=e.scorer)
        ax.set_title(title)
        ax.set_xticks(x, DEPTH_LABELS, rotation=30, ha="right", fontsize=8)
        ax.axhline(0.5, color="0.6", lw=0.8, ls="--")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
    axs[0, 0].set_ylabel("recall in top 1%")
    axs[1, 0].set_ylabel("recall in top 1%")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"recall in top 1% vs depth ({split}); null panel = false-positive rate")
    fig.tight_layout()
    fig.savefig(out / "compare_recall_vs_depth.png", dpi=110)
    plt.close(fig)
    L += ["## Plots", "", "![compare](compare_recall_vs_depth.png)", ""]
    (out / "report.md").write_text("\n".join(L))
    return names
