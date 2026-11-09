"""Anomaly scores from a trained encoder (ML_DESIGN §4) and §9 score files.

Sub-scores (all: higher = weirder):

* ``score_knn``: mean cosine distance to the k nearest *reference* stars (real train + val) in
  embedding space. The query's own TIC (or, for injections, its host TIC) is excluded from its
  neighbours, otherwise a val injection would find its own un-injected host.
* ``score_recon``: masked reconstruction surprise. Each patch is hidden in 2 of 5 block masks; per
  bin we take the *excess* NLL ``0.5 * (z^2 - 1)``, ``z = (y - mu) / sigma`` (the NLL minus its
  expectation under the model, so a star's overall noise level does not shift it). Star score =
  mean of the 4 most surprising patches (~32 h): localised events are not diluted.
  ``score_recon_mean`` (all bins) and ``score_recon_sap`` (SAP channel, top-4) are kept too.
* ``score_iforest``: Isolation Forest on embedding + scalar context.
* ``score``: rank-average of (knn, recon, iforest), each first converted to a percentile against
  the real stars, then re-expressed as a percentile within Tmag-bin x camera/CCD strata over the
  real stars. In [0, 1].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from oddball.common import strata as _strata
from oddball.ml.data import PATCH, Cache, iter_batches, make_inputs, scoring_masks

MIN_STRATUM = 300
CTX_WEIGHT = 2.0
COMBINE = ("score_knn", "score_recon", "score_iforest")


@torch.no_grad()
def encode_and_surprise(model, cache: Cache, rows, obs_mask=None, drop_sap=False, n_masks=5,
                        batch_size=256, seed=0, surprise_out=None, amp=True, log=print):  # fmt: skip
    """Pooled embeddings and reconstruction surprise for ``rows`` of ``cache``.

    ``surprise_out``: optional writable array [len(rows), L] (e.g. a memmap) that receives the
    time-resolved per-bin pdc surprise (NaN where no estimate)."""
    dev = next(model.parameters()).device
    P = model.cfg.n_patches
    om = None if obs_mask is None else torch.as_tensor(np.asarray(obs_mask), device=dev)
    pooled, rec_peak, rec_mean, rec_sap = [], [], [], []
    rng = np.random.default_rng(seed)
    done = 0
    for _, x, _ in iter_batches(cache, rows, batch_size, shuffle=False):
        xt = torch.from_numpy(x).to(dev)
        if drop_sap:
            xt[:, 1] = float("nan")
        inp, tgt, err, obs, sobs = make_inputs(xt, om)
        B = len(x)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            pooled.append(model.embed_pooled(inp).float().cpu().numpy())
            masks = np.stack([scoring_masks(P, n_masks, rng) for _ in range(B)], 1)  # [M,B,P]
            s_sum = torch.zeros(B, 2, inp.shape[1] * PATCH, device=dev)
            s_cnt = torch.zeros_like(s_sum)
            for mi in range(n_masks):
                m = torch.from_numpy(masks[mi]).to(dev)
                mu, logv, _ = model(inp, m)
                var = err[:, None, :] ** 2 + torch.nn.functional.softplus(logv.float()) + 1e-4
                exc = 0.5 * ((tgt - mu.float()) ** 2 / var - 1.0)
                mb = m.repeat_interleave(PATCH, 1)[:, None, :]
                w = (mb & torch.stack([obs, sobs], 1)).float()
                s_sum += exc * w
                s_cnt += w
        sur = torch.where(s_cnt > 0, s_sum / s_cnt.clamp(min=1), torch.full_like(s_sum, np.nan))
        # patch-level (>= 4 observed bins), then top-4 mean
        pc = (s_cnt > 0).float().view(B, 2, P, PATCH).sum(-1)
        ps = torch.nan_to_num(sur, nan=0.0).view(B, 2, P, PATCH).sum(-1) / pc.clamp(min=1)
        ps = torch.where(pc >= 4, ps, torch.full_like(ps, -1e9))
        top = torch.topk(ps, 4, dim=-1).values
        top = torch.where(top > -1e8, top, torch.full_like(top, np.nan))
        rec_peak.append(torch.nanmean(top[:, 0], -1).cpu().numpy())
        rec_sap.append(torch.nanmean(top[:, 1], -1).cpu().numpy())
        rec_mean.append(torch.nanmean(sur[:, 0], -1).cpu().numpy())
        if surprise_out is not None:
            surprise_out[done : done + B] = sur[:, 0].cpu().numpy().astype(surprise_out.dtype)
        done += B
        if done % (batch_size * 100) == 0:
            log(f"    encoded {done}/{len(rows)}")
    return (np.concatenate(pooled), np.concatenate(rec_peak), np.concatenate(rec_mean),
            np.concatenate(rec_sap))  # fmt: skip


def embedding_matrix(pooled: np.ndarray, ctx: np.ndarray, stats: dict) -> np.ndarray:
    """Final embedding = z-scored pooled tokens (d) + weighted z-scored scalar context (3)."""
    zp = (pooled - stats["p_mu"]) / stats["p_sd"]
    zc = (ctx - stats["c_mu"]) / stats["c_sd"]
    return np.concatenate([zp, CTX_WEIGHT * zc], 1).astype(np.float32)


def fit_embedding_stats(pooled_ref: np.ndarray, ctx_ref: np.ndarray) -> dict:
    return {"p_mu": pooled_ref.mean(0), "p_sd": pooled_ref.std(0) + 1e-6,
            "c_mu": ctx_ref.mean(0), "c_sd": ctx_ref.std(0) + 1e-6}  # fmt: skip


@torch.no_grad()
def knn_score(q: np.ndarray, q_tic: np.ndarray, ref: np.ndarray, ref_tic: np.ndarray, k: int = 20,
              chunk: int = 256, device=None) -> np.ndarray:  # fmt: skip
    """Mean cosine distance to the k nearest reference rows, excluding rows with the query's TIC.
    Falls back to CPU if the (shared) GPU runs out of memory."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            return _knn(q, q_tic, ref, ref_tic, k, chunk, device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            device = torch.device("cpu")
    return _knn(q, q_tic, ref, ref_tic, k, chunk, device)


def _knn(q, q_tic, ref, ref_tic, k, chunk, device):
    R = torch.nn.functional.normalize(torch.from_numpy(ref).to(device), dim=1)
    rt = torch.from_numpy(ref_tic.astype(np.int64)).to(device)
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), chunk):
        Q = torch.nn.functional.normalize(torch.from_numpy(q[i : i + chunk]).to(device), dim=1)
        qt = torch.from_numpy(q_tic[i : i + chunk].astype(np.int64)).to(device)
        sim = Q @ R.T
        vals, idx = torch.topk(sim, k + 3, dim=1)
        same = rt[idx] == qt[:, None]
        vals = torch.where(same, torch.full_like(vals, -2.0), vals)
        vals = torch.topk(vals, k, dim=1).values
        out[i : i + chunk] = (1.0 - vals).mean(1).cpu().numpy()
    return out


def tmag_bin(tmag: np.ndarray) -> np.ndarray:
    return _strata.tmag_bin(tmag)


def strata(camera, ccd, tmag) -> tuple[np.ndarray, np.ndarray]:
    """(fine, coarse) stratum keys: the shared Tmag-bin x camera/CCD id
    (``oddball.common.strata``, same definition as the evaluator), and Tmag bin alone."""
    return _strata.stratum_id(tmag, camera, ccd), _strata.tmag_bin(tmag)


def percentile_vs(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Mid-rank percentile of each x within the reference sample (NaN-safe: NaN -> 0.5)."""
    r = np.sort(ref[np.isfinite(ref)])
    if len(r) == 0:
        return np.full(len(x), 0.5)
    lo = np.searchsorted(r, x, "left")
    hi = np.searchsorted(r, x, "right")
    p = (lo + hi) / 2.0 / len(r)
    return np.where(np.isfinite(x), p, 0.5)


def stratified_percentile(x, fine_q, coarse_q, ref, fine_r, coarse_r, min_n=MIN_STRATUM):
    out = np.empty(len(x), dtype=np.float64)
    fine_counts = pd.Series(fine_r).value_counts()
    for key in np.unique(fine_q):
        sel = fine_q == key
        if fine_counts.get(key, 0) >= min_n:
            out[sel] = percentile_vs(ref[fine_r == key], x[sel])
        else:
            cq = coarse_q[sel]
            for c in np.unique(cq):
                s2 = np.flatnonzero(sel)[cq == c]
                rr = ref[coarse_r == c]
                out[s2] = percentile_vs(rr if len(rr) else ref, x[s2])
    return out


def combine(sub: pd.DataFrame, real_sub: pd.DataFrame, cols=COMBINE) -> np.ndarray:
    """Rank-average of ``cols`` + stratified percentile, both against the real stars
    (``real_sub``). With one column this is just its stratified percentile."""
    ra = np.mean([percentile_vs(real_sub[c].to_numpy(), sub[c].to_numpy()) for c in cols], 0)
    ra_real = np.mean([percentile_vs(real_sub[c].to_numpy(), real_sub[c].to_numpy())
                       for c in cols], 0)  # fmt: skip
    fq, cq = strata(sub.camera, sub.ccd, sub.tmag)
    fr, cr = strata(real_sub.camera, real_sub.ccd, real_sub.tmag)
    return ra, stratified_percentile(ra, fq, cq, ra_real, fr, cr)


def make_inputs_for(cache, rows, device, obs_mask=None):  # small helper for plots
    x = torch.from_numpy(np.asarray(cache.x[np.sort(rows)], dtype=np.float32)).to(device)
    om = None if obs_mask is None else torch.as_tensor(np.asarray(obs_mask), device=device)
    return make_inputs(x, om)


def obs_mask_for(ck: dict, sector: int, real_cache: Cache, work=None, log=print):
    """The common-gap mask for ``sector``: stored in the checkpoint if the model trained on that
    sector, else computed from the sector's real train + val stars (which bins have data; no
    labels) and cached as ``work/obs_mask_sNNNN.npy``. None if the model used no mask."""
    from pathlib import Path

    from oddball.ml.data import common_gap_masks

    om = ck.get("obs_masks") or {}
    if sector in om or str(sector) in om:
        return np.asarray(om.get(sector, om.get(str(sector))), dtype=bool)
    if not ck["train_cfg"].get("common_gap_mask"):
        return ck.get("obs_mask")
    f = None if work is None else Path(work) / f"obs_mask_s{sector:04d}.npy"
    if f is not None and f.exists():
        return np.load(f)
    idx = real_cache.index
    rows = idx.index[idx.split.isin(["train", "val"]) & idx.inj_id.isna()
                     & (idx.n_obs >= 200)].to_numpy()  # fmt: skip
    m = common_gap_masks(real_cache, rows)[0]
    if f is not None:
        np.save(f, m)
    log(f"  common-gap mask for held-out s{sector:04d}: {int(m.sum())} of {len(m)} bins")
    return m
