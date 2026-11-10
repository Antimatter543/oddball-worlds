"""Young-host (``yso1``) recall and matched-null report for several scorers (ML_DESIGN §15).

    uv run python -I scripts/yso_report.py --scorers classical ens_v1_eb ens_v2 ens_v3 \
        --split test --veto cmv3

Reads score files only (``oddball.eval.harness.load`` with ``--inj yso1``), so it uses the
evaluator's ranking rule: an injection is found when it is in the top 1% of the real stars of
its split and stratum after the symmetric veto. Per scorer:

* recall by depth bin and by kind, and overall with a 90% interval from a **host** bootstrap
  (each host carries ~20 replicas, so injections are not independent);
* the matched-null rate: unchanged copies of the hosts in the top 1% (count / n hosts);
* conditional recall: recall on hosts whose matched null is *not* already in the top 1%
  (what the injected event itself adds);
* recall vs injected SNR;
* a check that every matched null scores exactly like its host's real light curve (component
  percentiles), which tests the scoring pipeline end to end.

Writes ``outputs/eval/yso1/sNNNN/<scorers>__<split>[__veto-X]/report.md`` and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import outputs_root
from oddball.eval.harness import SNR_LABELS, load, score_dir
from oddball.score.ens3 import YSO_DEPTHS, cluster_bootstrap

COMPONENTS = ("classical", "ssl_v2_iforest", "ssl_v2_recon")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--scorers", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--veto", default="cmv3")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    cmd = "python scripts/yso_report.py " + " ".join(shlex.quote(x) for x in sys.argv[1:])
    sec = a.sector
    col = "top1%_veto" if a.veto != "none" else "top1%"
    summ, lines = {}, []
    by_depth, by_kind, by_snr = {}, {}, {}
    for s in a.scorers:
        ed = load(s, sec, "yso1", a.veto)
        q = ed.inj[ed.inj.split == a.split].copy()
        q["hit"] = q[col].fillna(False).astype(bool)
        astro = q[~q.is_null]
        nul = q[q.is_null]
        hosts_hit = set(nul.host_tic[nul.hit])
        cond = astro[~astro.host_tic.isin(hosts_hit)]
        rec, lo, hi = cluster_bootstrap(astro.hit.to_numpy(), astro.host_tic.to_numpy(),
                                        a.n_boot)  # fmt: skip
        dm = [float(astro.hit[astro.depth_bin == d].mean()) for d in YSO_DEPTHS]
        summ[s] = {
            "n_hosts": int(q.host_tic.nunique()), "n_inj": len(astro),
            "recall_all": rec, "recall_all_ci90": [lo, hi], "recall_mean_depths": float(np.mean(dm)),
            "recall_by_depth": dict(zip(YSO_DEPTHS, dm, strict=True)),
            "null_hits": int(nul.hit.sum()), "null_n": len(nul),
            "null_fpr": float(nul.hit.mean()) if len(nul) else float("nan"),
            "null_hit_tics": sorted(int(t) for t in hosts_hit),
            "cond_recall": float(cond.hit.mean()) if len(cond) else float("nan"),
            "cond_n_hosts": int(cond.host_tic.nunique()),
            "vetoed_frac": float(q.veto_flag.mean()) if "veto_flag" in q else 0.0,
        }  # fmt: skip
        by_depth[s] = astro.groupby("depth_bin", observed=True).hit.agg(["size", "mean"])
        by_kind[s] = astro.groupby("kind").hit.mean()
        by_snr[s] = astro.groupby("snr_bin", observed=False).hit.agg(["size", "mean"]).reindex(
            list(SNR_LABELS))

    # matched nulls must score exactly like the hosts' real light curves
    chk = {}
    t = ed.inj[["inj_id", "host_tic", "is_null"]]
    for c in COMPONENTS:
        sd = score_dir(c, sec)
        r = pd.read_parquet(sd / "real.parquet", columns=["tic", "score"])
        i = pd.read_parquet(sd / "inj_yso1.parquet", columns=["inj_id", "score"])
        j = t[t.is_null].merge(i, on="inj_id").merge(r.rename(columns={"tic": "host_tic"}),
                                                    on="host_tic", suffixes=("_null", "_real"))
        chk[c] = {"n": len(j), "max_abs_diff": float(np.abs(j.score_null - j.score_real).max())}

    lines += [f"# yso1 young-host injections, S{sec:02d}, {a.split} split, veto {a.veto}", "",
              f"Command: `{cmd}`", "",
              (f"Hosts: {summ[a.scorers[0]]['n_hosts']}; injections: "
              f"{summ[a.scorers[0]]['n_inj']}; matched nulls: {summ[a.scorers[0]]['null_n']}. "
              "Recall = in the top 1% of the real stars of the same split and stratum after the "
              "veto. Intervals: 90%, host bootstrap."), "",
              ("| scorer | recall (all) | mean over ~0.3/1/3/10% | null in top 1% | "
              "conditional recall (hosts) | vetoed |"), "|---|---|---|---|---|---|"]  # fmt: skip
    for s, m in summ.items():
        lines.append(f"| {s} | {m['recall_all']:.3f} [{m['recall_all_ci90'][0]:.3f}, "
                     f"{m['recall_all_ci90'][1]:.3f}] | {m['recall_mean_depths']:.3f} | "
                     f"{m['null_hits']}/{m['null_n']} ({m['null_fpr']:.3f}) | "
                     f"{m['cond_recall']:.3f} ({m['cond_n_hosts']}) | {m['vetoed_frac']:.3f} |")
    lines += ["", "Recall by depth bin (n):", "", "| depth | " + " | ".join(a.scorers) + " |",
              "|---|" + "---|" * len(a.scorers)]  # fmt: skip
    for d in YSO_DEPTHS:
        row = [f"{by_depth[s].loc[d, 'mean']:.3f}" if d in by_depth[s].index else "-"
               for s in a.scorers]  # fmt: skip
        n = int(by_depth[a.scorers[0]].loc[d, "size"]) if d in by_depth[a.scorers[0]].index else 0
        lines.append(f"| {d} ({n}) | " + " | ".join(row) + " |")
    lines += ["", "Recall by kind:", "", "| kind | " + " | ".join(a.scorers) + " |",
              "|---|" + "---|" * len(a.scorers)]  # fmt: skip
    for k in by_kind[a.scorers[0]].index:
        lines.append(f"| {k} | " + " | ".join(f"{by_kind[s][k]:.3f}" for s in a.scorers) + " |")
    lines += ["", "Recall vs injected SNR (n):", "", "| SNR | " + " | ".join(a.scorers) + " |",
              "|---|" + "---|" * len(a.scorers)]  # fmt: skip
    for b in SNR_LABELS:
        n = by_snr[a.scorers[0]].loc[b, "size"]
        row = [f"{by_snr[s].loc[b, 'mean']:.3f}" if by_snr[s].loc[b, "size"] else "-"
               for s in a.scorers]  # fmt: skip
        lines.append(f"| {b} ({0 if pd.isna(n) else int(n)}) | " + " | ".join(row) + " |")
    lines += ["", "Matched nulls vs the hosts' real light curves (component scores; must be 0):",
              "", *(f"- {c}: n {v['n']}, max |diff| {v['max_abs_diff']:.2e}"
                    for c, v in chk.items())]  # fmt: skip
    tag = "__".join(a.scorers) + f"__{a.split}" + ("" if a.veto == "cmv2" else f"__veto-{a.veto}")
    out = outputs_root() / "eval" / "yso1" / f"s{sec:04d}" / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps({"command": cmd, "scorers": summ,
                                                  "null_check": chk}, indent=1, default=float))
    print("\n".join(lines))
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
