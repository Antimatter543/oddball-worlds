"""GPU (torch/CUDA) coarse search: exact brute-force fold over the pre-registered period grid.

Same statistic as the CPU fold (search._fold): per phase bin sum N, sum D, epoch count and max
SES^2; valid bin needs count >= 2, sum N > 0 and max SES^2 <= DOMINANCE * MES^2. Unlike the CPU
path there is no FFA approximation: every period of period_grid(Plo, Phi, binw, T) is folded
exactly (this is the pre-registered grid rule).

Precision: sample positions u are exact integers (bin lattice). Phase = (u - n*p_int) - n*frac
with the integer part in int32 and only n*frac in float32, so the phase error is ~1e-4 bins
over a multi-year baseline. Sums are float32 (relative ~1e-6 at these sizes).

Use: search.search_star(pp, M, R, max_sig, coarse=GpuCoarse()).
"""

from __future__ import annotations

import math
import subprocess

import numpy as np
import torch
from search import DOMINANCE, period_grid

VRAM_BUDGET = 2.0e9  # bytes of temporaries per chunk; desktop keeps the rest of the 4 GB
BYTES_PER_ELEM = 44


def _flat_index_eager(u, uf, p_int, frac, P32, rows):
    """Global bin index (row * nbmax + phase bin) for each (period, sample)."""
    n = torch.floor(uf[None, :] / P32).to(torch.int32)  # may be off by one, fixed below
    ph = (u[None, :] - n * p_int).float() - n.float() * frac
    ph = torch.where(ph < 0, ph + P32, ph)
    ph = torch.where(ph >= P32, ph - P32, ph)
    b = torch.floor(ph + 1e-4).to(torch.int64)
    b = torch.minimum(b, p_int.to(torch.int64)).clamp(min=0)
    return b + rows


try:
    _flat_index = torch.compile(_flat_index_eager, dynamic=True)
except Exception:  # noqa: BLE001
    _flat_index = _flat_index_eager


def gpu_healthy() -> bool:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        return r.returncode == 0 and torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


class GpuCoarse:
    def __init__(self, device="cuda"):
        self.dev = torch.device(device)
        self.gpu_s = 0.0  # wall time spent in GPU folds (synchronised)

    @torch.no_grad()
    def __call__(self, dn, pr, binw, T):
        import time

        t0w = time.time()
        v = dn["cnt"] > 0
        u_np = np.flatnonzero(v).astype(np.int64)
        if len(u_np) < 3:
            return np.array([]), np.array([]), np.array([]), 0
        assert u_np.max() < 2**31 - 2**20
        dev = self.dev
        u = torch.from_numpy(u_np.astype(np.int32)).to(dev)
        uf = u.float()
        N = torch.from_numpy(dn["N"][v].astype(np.float32)).to(dev)
        D = torch.from_numpy(dn["D"][v].astype(np.float32)).to(dev)
        s2 = torch.where(N > 0, N * N / D, torch.zeros_like(N))
        src3 = torch.stack([N, D, torch.ones_like(N)], dim=1)
        last_pc = -1
        s2r = None
        ns = len(u_np)
        pg = period_grid(pr[0], pr[1], binw, T)
        pu_all = pg / binw  # periods in samples (float64 on host)
        best_m = np.zeros(len(pg))
        best_b = np.full(len(pg), -1, np.int64)
        chunk = max(1, int(VRAM_BUDGET / (BYTES_PER_ELEM * ns)))
        for c0 in range(0, len(pg), chunk):
            pu = pu_all[c0:c0 + chunk]
            pc = len(pu)
            nbmax = int(math.floor(pu.max())) + 1
            if c0 == 0 or pc != last_pc:
                s2r = s2.repeat(pc)
                last_pc = pc
            p_int_np = np.floor(pu).astype(np.int32)
            p_int = torch.from_numpy(p_int_np).to(dev)[:, None]
            frac = torch.from_numpy((pu - p_int_np).astype(np.float32)).to(dev)[:, None]
            P32 = torch.from_numpy(pu.astype(np.float32)).to(dev)[:, None]
            rows = torch.arange(pc, device=dev, dtype=torch.int64)[:, None] * nbmax
            flat = _flat_index(u, uf, p_int, frac, P32, rows).view(-1)
            size = pc * nbmax
            # one atomic pass for sumN, sumD, count
            acc = torch.zeros(size, 3, device=dev).index_add_(0, flat, src3.repeat(pc, 1))
            aN, aD, cnt = acc[:, 0], acc[:, 1], acc[:, 2]
            m2 = torch.where(aD > 0, aN * aN / aD.clamp(min=1e-30), torch.zeros_like(aN))
            # dominance without a float atomic max: a bin is invalid iff some sample in it has
            # SES^2 > DOMINANCE * MES^2 (identical to max SES^2 > DOMINANCE * MES^2)
            viol = s2r > DOMINANCE * m2[flat]
            bad = torch.zeros(size, dtype=torch.bool, device=dev)
            bad[flat[viol]] = True
            del flat, viol, acc
            col = torch.arange(nbmax, device=dev)[None, :]
            ok = ((cnt >= 2) & (aN > 0) & ~bad).view(pc, nbmax) & (col <= p_int)
            m2 = torch.where(ok, m2.view(pc, nbmax), torch.zeros(1, device=dev))
            bm, bb = m2.max(dim=1)
            bb = torch.where(bm > 0, bb, torch.full_like(bb, -1))
            best_m[c0:c0 + pc] = torch.sqrt(bm).double().cpu().numpy()
            best_b[c0:c0 + pc] = bb.cpu().numpy()
            del aN, aD, cnt, m2, ok, bad
        torch.cuda.synchronize()
        self.gpu_s += time.time() - t0w
        ok = best_b >= 0
        t0 = dn["t_start"] + (best_b[ok] + 0.5) * binw
        return pg[ok], best_m[ok], t0, int(ns) * len(pg)
