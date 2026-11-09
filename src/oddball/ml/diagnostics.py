"""Does the masked reconstruction use context? (ML_DESIGN §3 sanity checks, §10 ssl_v1 flaw.)

Two checks, both cheap enough to log every epoch:

* ``masked_errors``: on hidden, observed PDC bins, the model's squared error against two
  baselines that use no learning: predicting the star's median (0 in the shape view) and linear
  interpolation across the hidden block from the nearest visible bins. On *signal-dominated*
  stars (``data.signal_dominated``: scatter mostly stellar variability) a model that uses
  context must beat the median by a clear margin. ``gain_vs_zero = 1 - mse_model / mse_zero``.
* ``sinusoid_test``: clean synthetic sinusoids (1% amplitude, 0.2% white noise) put through the
  same standardisation as real stars; one patch (8 h) or a 4-patch block (32 h) is hidden and
  the Pearson correlation between prediction and truth inside it is measured. A model that
  predicts the median everywhere scores ~0.
"""

from __future__ import annotations

import numpy as np
import torch

from oddball.ml.data import GRID_LEN, PATCH, make_inputs, standardise
from oddball.ml.model import patch_to_bin


def interp_baseline(y: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of ``y`` [B,L] across bins where ``vis`` [B,L] is False, from the
    nearest visible bins on each side (constant extrapolation at the ends)."""
    B, L = y.shape
    pos = torch.arange(L, device=y.device).expand(B, L)
    left = torch.where(vis, pos, torch.full_like(pos, -1)).cummax(1).values
    right = torch.where(vis, pos, torch.full_like(pos, L)).flip(1).cummin(1).values.flip(1)
    has_l, has_r = left >= 0, right < L
    yl = torch.gather(y, 1, left.clamp(0, L - 1))
    yr = torch.gather(y, 1, right.clamp(0, L - 1))
    w = (pos - left).float() / (right - left).clamp(min=1).float()
    both = yl + w * (yr - yl)
    out = torch.where(has_l & has_r, both, torch.where(has_l, yl, torch.where(has_r, yr, 0 * y)))
    return torch.where(vis, y, out)


@torch.no_grad()
def masked_errors(model, inp, tgt, err, obs, mask) -> np.ndarray:
    """Per-star sums over hidden observed PDC bins: [B, 4] = (n, sse_model, sse_zero, sse_interp)."""
    mu, _logv, _ = model(inp, mask)
    mb = patch_to_bin(mask) & obs
    y = tgt[:, 0]
    yi = interp_baseline(y, obs & ~patch_to_bin(mask))
    w = mb.float()
    out = torch.stack([w.sum(1), ((y - mu[:, 0].float()) ** 2 * w).sum(1), (y**2 * w).sum(1),
                       ((y - yi) ** 2 * w).sum(1)], 1)  # fmt: skip
    return out.cpu().numpy().astype(np.float64)


def summarise(sums: np.ndarray) -> dict:
    n, sm, s0, si = sums.sum(0)
    n = max(n, 1.0)
    return {"n_bins": int(n), "mse_model": sm / n, "mse_zero": s0 / n, "mse_interp": si / n,
            "gain_vs_zero": 1 - sm / max(s0, 1e-12), "gain_vs_interp": 1 - sm / max(si, 1e-12)}


def synthetic_sinusoids(n: int, period_d: float = 0.7, amp: float = 0.01, noise: float = 0.002,
                        seed: int = 0) -> np.ndarray:
    """[n,3,L] shape-view inputs (pdc, sap, err) of clean sinusoids with random phase, observed
    on bins 20..1220 with a 1-d mid-sector gap, standardised exactly like real stars."""
    rng = np.random.default_rng(seed)
    t = np.arange(GRID_LEN) * 30 / 1440
    ph = rng.uniform(0, 2 * np.pi, (n, 1))
    f = 1 + amp * np.sin(2 * np.pi * t / period_d + ph) + noise * rng.standard_normal((n, GRID_LEN))
    f[:, :20] = np.nan
    f[:, 1220:] = np.nan
    f[:, 600:648] = np.nan
    e = np.where(np.isfinite(f), noise, np.nan)
    st = standardise(f, e, f.copy())
    return np.stack([st["pdc"], st["sap"], st["err"]], 1).astype(np.float32)


@torch.no_grad()
def sinusoid_test(model, device, drop_sap: bool, n: int = 64, period_d: float = 0.7,
                  seed: int = 0) -> dict:  # fmt: skip
    """Correlation between prediction and truth inside a hidden 1-patch and 4-patch block."""
    x = torch.from_numpy(synthetic_sinusoids(n, period_d, seed=seed)).to(device)
    if drop_sap:
        x[:, 1] = float("nan")
    inp, tgt, _err, obs, _ = make_inputs(x)
    P = GRID_LEN // PATCH
    rng = np.random.default_rng(seed + 1)
    res = {}
    for width in (1, 4):
        cs, pool_y, pool_p = [], [], []
        for _rep in range(4):
            m = np.zeros((n, P), dtype=bool)
            for b in range(n):
                while True:  # a block fully inside observed data
                    s = int(rng.integers(2, P - width - 2))
                    if obs[b, s * PATCH : (s + width) * PATCH].all():
                        break
                m[b, s : s + width] = True
            mt = torch.from_numpy(m).to(device)
            mu, _, _ = model(inp, mt)
            hb = patch_to_bin(mt) & obs
            for b in range(n):
                yy = tgt[b, 0][hb[b]].cpu().numpy()
                pp = mu[b, 0][hb[b]].float().cpu().numpy()
                pool_y.append(yy)
                pool_p.append(pp)
                if pp.std() > 1e-6 and yy.std() > 1e-6:
                    cs.append(np.corrcoef(yy, pp)[0, 1])
                else:
                    cs.append(0.0)
        yy, pp = np.concatenate(pool_y), np.concatenate(pool_p)
        res[f"hide{width}"] = {
            "corr_mean": float(np.mean(cs)),
            "corr_pooled": float(np.corrcoef(yy, pp)[0, 1]) if pp.std() > 1e-6 else 0.0,
            "mse_model": float(((yy - pp) ** 2).mean()), "mse_zero": float((yy**2).mean()),
        }  # fmt: skip
    return res
