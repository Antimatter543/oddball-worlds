"""Visual sanity plots: one example injection per kind (host before/after, PDCSAP and SAP).

    ODDBALL_DATA=... uv run python -I scripts/plot_injections.py --name v1 --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import outputs_root
from oddball.inject.build import injection_dir
from oddball.tess import store


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--name", default="v1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--min-depth", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    d = injection_dir(a.name, a.sector, a.split)
    inj = pd.read_parquet(d / "injections.parquet")
    rng = np.random.default_rng(a.seed)
    picks = []
    for kind, g in inj.groupby("kind"):
        g = g[(g.depth >= a.min_depth) & g.peak_bin_valid]
        if len(g):
            picks.append(g.iloc[rng.integers(len(g))])
    picks = pd.DataFrame(picks)
    tbl = ds.dataset(sorted((d / "shards").glob("lc_*.parquet"))).to_table(filter=ds.field("inj_id").isin(picks.inj_id.tolist()))
    rows = {r["inj_id"]: r for r in tbl.to_pylist()}
    hosts = store.load_lightcurves(a.sector, picks.host_tic.tolist())

    fig, axs = plt.subplots(len(picks), 1, figsize=(13, 2.3 * len(picks)))
    for ax, p in zip(np.atleast_1d(axs), picks.itertuples(), strict=True):
        h = hosts[int(p.host_tic)]
        r = rows[p.inj_id]
        t = h["time"]
        ax.plot(t, h["flux"], ",", color="0.6", label="host PDCSAP")
        ax.plot(t, np.asarray(r["flux"]), ".", ms=2, color="tab:blue", label="injected PDCSAP")
        ax.plot(t, np.asarray(r["sap_flux"]) - 3 * p.depth, ".", ms=2, color="tab:orange",
                label="injected SAP (offset)")
        ax.axvline(p.t0_btjd, color="tab:red", lw=0.6)
        pj = json.loads(p.params_json)
        ax.set_title(f"{p.kind}: depth {p.depth:.3%}, dur {p.duration_d * 24:.1f} h, "
                     f"T={p.tmag:.1f}, CROWDSAP={p.crowdsap:.2f}, n_comp={len(pj['components'])}",
                     fontsize=9)
        ax.legend(fontsize=7, loc="lower left", markerscale=4)
    fig.tight_layout()
    out = outputs_root() / "eval" / "injections"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"examples_{a.name}_s{a.sector:04d}_{a.split}.png"
    fig.savefig(f, dpi=100)
    print(f)


if __name__ == "__main__":
    main()
