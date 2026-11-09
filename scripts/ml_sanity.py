"""Required sanity checks before trusting a run (ML_DESIGN §3).

    ODDBALL_DATA=... uv run python -I scripts/ml_sanity.py --run runs/ssl_v1_s0 --all
    ODDBALL_DATA=... uv run python -I scripts/ml_sanity.py --run runs/sanity_overfit64 --overfit

* ``--overfit``: on a tiny-batch run, masked chi^2 (mean z^2) vs the noise floor (~1).
* ``--copy``: masked-bin MSE of the model vs a "copy the neighbours" baseline (linear interpolation
  across the hidden block from the nearest visible bins) and vs predicting the median (0). If the
  model is no better than interpolation, masking is too easy / it learned to copy.
* ``--probe``: nuisance probe (camera, camera/CCD, Tmag bin) on the trained encoder vs chance.
* ``--plots``: reconstructions for random, faint, bright and known-weird stars.
Results go to ``runs/<id>/sanity.json`` and ``outputs/ml/<scorer>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from oddball.common.paths import REPO_ROOT, outputs_root, sector_dir
from oddball.ml.data import (
    PATCH,
    Cache,
    cache_dir_for,
    contiguous_block_mask,
    make_inputs,
    training_rows,
)
from oddball.ml.model import patch_to_bin
from oddball.ml.probe import run_probes
from oddball.ml.train import load_model


def batch_inputs(cache, rows, ck, dev):
    x = torch.from_numpy(np.asarray(cache.x[np.sort(rows)], dtype=np.float32)).to(dev)
    om = ck.get("obs_mask")
    if ck["train_cfg"].get("drop_sap"):
        x[:, 1] = float("nan")
    return make_inputs(x, None if om is None else torch.as_tensor(np.asarray(om), device=dev))


@torch.no_grad()
def masked_stats(model, ck, cache, rows, dev, seed=0, bs=256):
    """mean z^2, model MSE, interpolation MSE, zero MSE over masked observed pdc bins."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(4)
    n = 0
    for i in range(0, len(rows), bs):
        r = np.sort(rows[i : i + bs])
        inp, tgt, err, obs, _sobs = batch_inputs(cache, r, ck, dev)
        m = torch.from_numpy(contiguous_block_mask(len(r), model.cfg.n_patches, 0.4, rng)).to(dev)
        mu, logv, _ = model(inp, m)
        var = err**2 + torch.nn.functional.softplus(logv[:, 0]) + 1e-4
        mb = patch_to_bin(m) & obs
        y = tgt[:, 0]
        # interpolation baseline across hidden bins, from visible observed bins only
        vis = (obs & ~patch_to_bin(m)).cpu().numpy()
        yn = y.cpu().numpy()
        interp = np.zeros_like(yn)
        grid = np.arange(yn.shape[1])
        for b in range(len(r)):
            v = vis[b]
            if v.sum() >= 2:
                interp[b] = np.interp(grid, grid[v], yn[b, v])
        interp = torch.from_numpy(interp).to(dev)
        w = mb.float()
        acc += np.array([
            float((((y - mu[:, 0]) ** 2 / var) * w).sum()),
            float((((y - mu[:, 0]) ** 2) * w).sum()),
            float((((y - interp) ** 2) * w).sum()),
            float(((y**2) * w).sum()),
        ])  # fmt: skip
        n += float(w.sum())
    z2, mse, mse_i, mse_0 = acc / max(n, 1)
    return {"mean_z2": round(z2, 4), "mse_model": round(mse, 4), "mse_interp": round(mse_i, 4),
            "mse_zero": round(mse_0, 4), "n_bins": int(n)}  # fmt: skip


@torch.no_grad()
def pooled_for(model, ck, cache, rows, dev, bs=500):
    out = []
    for i in range(0, len(rows), bs):
        inp, *_ = batch_inputs(cache, rows[i : i + bs], ck, dev)
        out.append(model.embed_pooled(inp).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def plot_recon(model, ck, cache, rows, labels, path: Path, dev, seed=1):
    rows = np.asarray(rows)
    order = np.argsort(rows)
    rows, labels = rows[order], [labels[i] for i in order]
    inp, tgt, err, obs, sobs = batch_inputs(cache, rows, ck, dev)
    m = torch.from_numpy(contiguous_block_mask(len(rows), model.cfg.n_patches, 0.4,
                                               np.random.default_rng(seed))).to(dev)  # fmt: skip
    mu, logv, _ = model(inp, m)
    var = err[:, None] ** 2 + torch.nn.functional.softplus(logv) + 1e-4
    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.1 * n), squeeze=False)
    t = np.arange(tgt.shape[-1]) * 30 / 1440
    for i, ax in enumerate(axes[:, 0]):
        o = obs[i].cpu().numpy()
        mb = patch_to_bin(m)[i].cpu().numpy()
        y = tgt[i, 0].cpu().numpy()
        s = tgt[i, 1].cpu().numpy()
        so = sobs[i].cpu().numpy()
        p = mu[i, 0].cpu().numpy()
        sd = np.sqrt(var[i, 0].cpu().numpy())
        for j in range(model.cfg.n_patches):
            if m[i, j]:
                ax.axvspan(t[j * PATCH], t[min(len(t) - 1, (j + 1) * PATCH)], color="0.9", lw=0)
        ax.plot(t[so], s[so] - 8, ".", ms=1.5, color="tab:orange", label="SAP (offset -8)")
        ax.plot(t[o], y[o], ".", ms=1.5, color="k", label="PDC (shape view)")
        pm = np.where(mb & o, p, np.nan)
        ax.plot(t, pm, "-", color="tab:red", lw=1, label="prediction (hidden)")
        ax.fill_between(t, pm - sd, pm + sd, color="tab:red", alpha=0.2, lw=0)
        r = cache.index.loc[rows[i]]
        ax.set_title(f"{labels[i]}  TIC {int(r.tic)}  Tmag {r.tmag:.1f}  cam{int(r.camera)}/"
                     f"ccd{int(r.ccd)}  split={r.split}", fontsize=8, loc="left")  # fmt: skip
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right", ncol=3)
    axes[-1, 0].set_xlabel("days from sector grid start")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def overfit_fixed(cache, dev, n_rows=32, steps=1500, lr=1e-3, obs_mask=None) -> dict:
    """Fresh model, one fixed batch, one fixed mask: masked loss and chi^2 must collapse."""
    from oddball.ml.model import ModelConfig, PatchMAE, gaussian_nll

    torch.manual_seed(0)
    model = PatchMAE(ModelConfig()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    rows = training_rows(cache, "train")[:n_rows]
    ck = {"obs_mask": obs_mask, "train_cfg": {}}
    inp, tgt, err, obs, sobs = batch_inputs(cache, rows, ck, dev)
    m = torch.from_numpy(contiguous_block_mask(len(rows), model.cfg.n_patches, 0.4,
                                               np.random.default_rng(0))).to(dev)  # fmt: skip
    hist = []
    for s in range(steps):
        mu, logv, _ = model(inp, m)
        loss = gaussian_nll(mu, logv, tgt, err, obs, sobs, patch_to_bin(m))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 250 == 0 or s == steps - 1:
            with torch.no_grad():
                var = err**2 + torch.nn.functional.softplus(logv[:, 0]) + 1e-4
                w = (patch_to_bin(m) & obs).float()
                z2 = float((((tgt[:, 0] - mu[:, 0]) ** 2 / var) * w).sum() / w.sum())
            hist.append({"step": s, "loss": round(float(loss), 4), "mean_z2": round(z2, 4)})
    return {"n_rows": n_rows, "steps": steps, "history": hist}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overfit-fixed", action="store_true")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scorer", default="ssl_v1")
    ap.add_argument("--sector", type=int, default=32)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    run = a.run if a.run.is_absolute() else REPO_ROOT / a.run
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(run / "best.pt", dev)
    cache = Cache(cache_dir_for(sector_dir(a.sector)))
    out_json = run / "sanity.json"
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    rng = np.random.default_rng(0)
    idx = cache.index

    if a.overfit_fixed:
        res["overfit_fixed"] = overfit_fixed(cache, dev, obs_mask=ck.get("obs_mask"))
        print("overfit fixed", json.dumps(res["overfit_fixed"]))
    if a.overfit:
        rows = training_rows(cache, "train")[: ck["train_cfg"]["overfit_rows"]]
        res["overfit"] = masked_stats(model, ck, cache, rows, dev)
        print("overfit", res["overfit"])
    if a.copy or a.all:
        va = np.sort(rng.choice(training_rows(cache, "val"), 4000, replace=False))
        res["copy_check_val"] = masked_stats(model, ck, cache, va, dev)
        print("copy check (val)", res["copy_check_val"])
    if a.probe or a.all:
        tr = np.sort(rng.choice(training_rows(cache, "train"), 8000, replace=False))
        va = np.sort(rng.choice(training_rows(cache, "val"), 4000, replace=False))
        ztr = pooled_for(model, ck, cache, tr, dev)
        zva = pooled_for(model, ck, cache, va, dev)
        res["nuisance_probe"] = run_probes(ztr, idx.loc[tr], zva, idx.loc[va])
        print("probe", json.dumps(res["nuisance_probe"]))
    if a.plots or a.all:
        od = outputs_root() / "ml" / a.scorer
        real = idx.index[idx.inj_id.isna() & (idx.split != "known")].to_numpy()
        faint = idx.index[(idx.tmag > 14.5) & (idx.split == "val")].to_numpy()
        bright = idx.index[(idx.tmag < 8.5) & (idx.split == "val")].to_numpy()
        known = idx.index[idx.split == "known"].to_numpy()
        paths = []
        for name, pool in (("random", real), ("faint", faint), ("bright", bright)):
            pick = rng.choice(pool, min(6, len(pool)), replace=False)
            paths.append(plot_recon(model, ck, cache, pick, [name] * len(pick),
                                    od / f"recon_{name}.png", dev))  # fmt: skip
        for j in range(0, len(known), 6):
            pick = known[j : j + 6]
            paths.append(plot_recon(model, ck, cache, pick, ["known"] * len(pick),
                                    od / f"recon_known_{j // 6}.png", dev))  # fmt: skip
        res["recon_plots"] = paths
        print("plots", paths)
    out_json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
