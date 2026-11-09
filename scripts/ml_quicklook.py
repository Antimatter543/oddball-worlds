"""Quick look at a scored run: UMAP, blind known-positive percentiles, top-N gallery, seed overlap.

    ODDBALL_DATA=... uv run python -I scripts/ml_quicklook.py --run runs/ssl_v1_s0 --scorer ssl_v1 \
        [--compare ssl_v1_seed1 ssl_v1_seed2]

Known positives are only *reported* here (blind check). Nothing is tuned on them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oddball.common.paths import REPO_ROOT, outputs_root
from oddball.common.splits import KNOWN_POSITIVES_CSV
from oddball.tess.store import load_lightcurves


def known_table(real: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    kp = pd.read_csv(KNOWN_POSITIVES_CSV)[["tic", "name", "class"]].drop_duplicates("tic")
    k = sub[sub.split == "known"].merge(kp, on="tic", how="left")
    n = len(real)
    order = real.score.rank(ascending=False, method="min")
    rank = dict(zip(real.tic, order, strict=True))
    cols = ["score_knn", "score_recon", "score_iforest", "score_recon_sap"]
    out = k[["tic", "name", "class", "tmag", "camera", "ccd"]].copy()
    out["score_pct"] = (100 * k.score.to_numpy()).round(2)
    out["global_rank"] = [int(rank[t]) for t in k.tic]
    out["global_top_pct"] = (100 * out.global_rank / n).round(2)
    for c in cols:  # global percentile of each sub-score among real stars
        ref = np.sort(real[c].to_numpy())
        out[c + "_pct"] = (100 * np.searchsorted(ref, k[c].to_numpy()) / n).round(1)
    return out.sort_values("global_rank")


def plot_gallery(top: pd.DataFrame, tr: pd.DataFrame, origin: int, sector: int, path: Path):
    lcs = load_lightcurves(sector, top.tic.tolist())
    n = len(top)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.0 * n), squeeze=False)
    trd = {int(r.tic): r.surprise for r in tr.itertuples()}
    for ax, r in zip(axes[:, 0], top.itertuples(), strict=True):
        lc = lcs.get(int(r.tic))
        if lc is not None:
            ax.plot(lc["time"], lc["flux"], ".", ms=1.5, color="k", label="PDCSAP")
            ax.plot(lc["time"], lc["sap_flux"] - 0, ".", ms=1.0, color="tab:orange", alpha=0.5,
                    label="SAP")  # fmt: skip
            f = lc["flux"][np.isfinite(lc["flux"])]
            if len(f):
                lo, hi = np.percentile(f, [0.2, 99.8])
                pad = 0.3 * (hi - lo) + 1e-4
                ax.set_ylim(lo - pad, hi + pad)
        s = trd.get(int(r.tic))
        if s is not None:
            t = (origin + np.arange(len(s)) + 0.5) * 30 / 1440
            ax2 = ax.twinx()
            ax2.fill_between(t, 0, np.nan_to_num(np.clip(s, 0, None)), color="tab:red", alpha=0.25,
                             lw=0)  # fmt: skip
            ax2.set_yticks([])
        ax.set_title(f"#{r.rank} TIC {int(r.tic)}  Tmag {r.tmag:.1f} cam{int(r.camera)}/ccd"
                     f"{int(r.ccd)}  score {r.score:.4f}  knn {r.score_knn:.3f} recon "
                     f"{r.score_recon:.1f} iforest {r.score_iforest:.3f}", fontsize=8, loc="left")
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=6, loc="upper right")
    axes[-1, 0].set_xlabel("BTJD  (red = time-resolved reconstruction surprise)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def umap_plot(emb: np.ndarray, sub: pd.DataFrame, path: Path, n: int = 30000, seed: int = 0):
    import umap

    rng = np.random.default_rng(seed)
    real = np.flatnonzero((sub.split != "known").to_numpy())
    pick = np.r_[np.sort(rng.choice(real, min(n, len(real)), replace=False)),
                 np.flatnonzero((sub.split == "known").to_numpy())]  # fmt: skip
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=seed,
                   low_memory=True).fit_transform(emb[pick].astype(np.float32))  # fmt: skip
    s = sub.iloc[pick]
    kp = pd.read_csv(KNOWN_POSITIVES_CSV)[["tic", "class"]].drop_duplicates("tic")
    cls = s[["tic"]].merge(kp, on="tic", how="left")["class"].to_numpy()
    isk = (s.split == "known").to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(19, 6))
    sc = axes[0].scatter(xy[:, 0], xy[:, 1], c=s.tmag, s=0.5, cmap="viridis", vmin=7, vmax=15)
    fig.colorbar(sc, ax=axes[0], label="Tmag")
    cc = (s.camera * 4 + s.ccd - 5).to_numpy()
    axes[1].scatter(xy[:, 0], xy[:, 1], c=cc, s=0.5, cmap="tab20", vmin=0, vmax=19)
    axes[1].set_title("camera/CCD (16 colours)")
    axes[2].scatter(xy[~isk, 0], xy[~isk, 1], c="0.8", s=0.5)
    for c in sorted(set(cls[isk].astype(str))):
        m = isk & (cls.astype(str) == c)
        axes[2].scatter(xy[m, 0], xy[m, 1], s=40, marker="*", label=f"{c} ({m.sum()})")
    axes[2].legend(fontsize=8)
    axes[2].set_title("known positives (blind)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("Tmag")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def top_overlap(a: pd.DataFrame, b: pd.DataFrame, n: int = 100) -> float:
    ta = set(a.nlargest(n, "score").tic)
    tb = set(b.nlargest(n, "score").tic)
    return len(ta & tb) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scorer", default="ssl_v1")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--compare", nargs="*", default=[])
    ap.add_argument("--no-umap", action="store_true")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else REPO_ROOT / a.run
    sdir = f"s{a.sector:04d}"
    od = outputs_root() / "ml" / a.scorer
    real = pd.read_parquet(outputs_root() / "scores" / a.scorer / sdir / "real.parquet")
    sub = pd.read_parquet(run / "score" / f"real_sub_{sdir}.parquet")
    origin = int(__import__("torch").load(run / "best.pt", map_location="cpu",
                                          weights_only=False)["origin"])  # fmt: skip
    res = {}

    kt = known_table(real, sub)
    kt.to_csv(od / f"known_percentiles_{sdir}.csv", index=False)
    res["known"] = kt.to_dict("records")
    res["known_in_top1pct"] = int((kt.global_top_pct <= 1).sum())
    res["known_n"] = len(kt)
    print(kt.to_string(index=False))

    nk = sub[sub.split != "known"].sort_values("score", ascending=False, kind="stable")
    nk = nk.assign(rank=np.arange(1, len(nk) + 1))
    top = nk.head(a.top)
    top.drop(columns=[c for c in top if c.startswith("_")]).to_csv(od / f"top{a.top}_{sdir}.csv",
                                                                  index=False)  # fmt: skip
    tr = pd.read_parquet(od / f"timeresolved_{sdir}_real.parquet")
    half = (len(top) + 1) // 2
    for j, part in enumerate((top.iloc[:half], top.iloc[half:])):
        plot_gallery(part, tr, origin, a.sector, od / f"top{a.top}_{sdir}_{j}.png")
    res["top"] = top[["rank", "tic", "tmag", "camera", "ccd", "split", "score", "score_knn",
                      "score_recon", "score_iforest"]].to_dict("records")  # fmt: skip
    res["top_ccd_counts"] = (nk.head(200).camera * 10 + nk.head(200).ccd).value_counts().to_dict()
    res["top_tmag_median"] = float(nk.head(200).tmag.median())
    print(top[["rank", "tic", "tmag", "camera", "ccd", "score_knn", "score_recon",
               "score_iforest"]].to_string(index=False))  # fmt: skip

    if a.compare:
        scorers = [a.scorer] + a.compare
        files = {s: pd.read_parquet(outputs_root() / "scores" / s / sdir / "real.parquet")
                 for s in scorers}  # fmt: skip
        res["top100_overlap"] = {f"{x}|{y}": top_overlap(files[x], files[y])
                                 for i, x in enumerate(scorers) for y in scorers[i + 1:]}  # fmt: skip
        for c in ("score_knn", "score_recon", "score_iforest"):
            res[f"top100_overlap_{c}"] = {
                f"{x}|{y}": top_overlap(files[x].rename(columns={"score": "_s", c: "score"}),
                                        files[y].rename(columns={"score": "_s", c: "score"}))
                for i, x in enumerate(scorers) for y in scorers[i + 1:]}  # fmt: skip
        print("top-100 overlap", json.dumps({k: v for k, v in res.items() if "overlap" in k}))

    if not a.no_umap:
        emb = np.load(run / "score" / f"emb_real_{sdir}.npy")
        umap_plot(emb, sub, od / f"umap_{sdir}.png")
        res["umap"] = str(od / f"umap_{sdir}.png")
    (od / f"quicklook_{sdir}.json").write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
