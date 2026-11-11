"""All-sector box search for one star: preprocessing, SES, fold over a baseline-aware grid.

Method frozen in docs/spikes/mdwarf_stack.md (pre-registration). Summary:
  raw 10-min binned PDCSAP -> trim segment edges -> flare clip (upward only) -> biweight 0.5 d
  -> per-duration box SES with per-sector red-noise rescaling -> fold over 1-60 d with
  dP/P = stride*dt/T_base -> valid peak needs >=2 epochs and single-event dominance <= 0.8
  -> refine -> mask +-1 D -> iterate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from common import BIN_D
from numba import njit

G_AU3_MSUN_D2 = 2.959122082855911e-4  # G in AU^3 / (M_sun d^2)
RSUN_AU = 0.00465047

# bins of 10 min: 1-4 h. Amendment 1 (compute, before any null/host result): the pre-registered
# 0.5 h and 0.67 h boxes are dropped; at stride 1 they cost >2/3 of the search.
# Amendment 2 (compute, after timing two real controls, before any null/host run): 5 boxes
# spaced ~1.4x instead of 7, and coarse stride D/2. Refinement stays at stride 1.
DURATIONS = (6, 9, 13, 18, 24)
# FFA phase error ~ T*binw/P^2 samples (row skips when the drift exceeds a period). Below
# P_sw = sqrt(FFA_ERR_MAX * T * binw) the exact brute fold is used instead.
FFA_ERR_MAX = 1.0
DUR_RATIO = (0.33, 1.5)  # allowed D / D_c(P)
P_MIN, P_MAX = 1.0, 60.0
GAP_D, TRIM_D = 0.5, 0.25
WINDOW_D = 0.5
FLARE_SIG, FLARE_TAIL = 3.0, 3
MIN_COVER = 0.5
DOMINANCE = 0.8
MES_FLOOR = 6.0


# --------------------------------------------------------------------------------------------
# stellar geometry


def central_duration(P, mass, radius):
    """Central (b=0), circular transit duration in days."""
    a = (G_AU3_MSUN_D2 * mass * P**2 / (4 * math.pi**2)) ** (1 / 3)
    x = np.clip(radius * RSUN_AU / a, 0, 1)
    return P / math.pi * np.arcsin(x)


def period_range_for(k: int, mass: float, radius: float) -> tuple[float, float] | None:
    """Periods in [P_MIN, P_MAX] where D_k / D_c(P) lies in DUR_RATIO."""
    D = k * BIN_D
    Ps = np.geomspace(P_MIN, P_MAX, 2000)
    r = D / central_duration(Ps, mass, radius)
    ok = (r >= DUR_RATIO[0]) & (r <= DUR_RATIO[1])
    if not ok.any():
        return None
    return float(Ps[ok].min()), float(Ps[ok].max())


# --------------------------------------------------------------------------------------------
# preprocessing


def robust_sigma(x):
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _segments(idx):
    br = np.flatnonzero(np.diff(idx) * BIN_D > GAP_D) + 1
    return np.split(np.arange(len(idx)), br)


def _per_sector_sigma(resid, sector):
    sig = np.empty_like(resid)
    for s in np.unique(sector):
        m = sector == s
        sig[m] = robust_sigma(resid[m])
    return sig


def preprocess(lc: dict) -> dict:
    """lc: idx, flux (raw normalised), sector. Returns idx, flux (detrended), sigma, sector."""
    from wotan import flatten

    idx, flux, sector = lc["idx"], lc["flux"].astype(np.float64), lc["sector"]
    keep = np.isfinite(flux)
    # 1. trim segment edges
    for seg in _segments(idx):
        t = idx[seg] * BIN_D
        keep[seg[(t < t[0] + TRIM_D) | (t > t[-1] - TRIM_D)]] = False
    idx, flux, sector = idx[keep], flux[keep], sector[keep]
    t = (idx + 0.5) * BIN_D
    # 2. flare clip, upward only, twice
    for _ in range(2):
        trend = flatten(t, flux, method="biweight", window_length=WINDOW_D, return_trend=True,
                        break_tolerance=GAP_D, edge_cutoff=0)[1]
        resid = flux / trend - 1
        sig = _per_sector_sigma(resid, sector)
        hi = np.flatnonzero(resid > FLARE_SIG * sig)
        bad = np.zeros(len(t), bool)
        for j in range(FLARE_TAIL + 1):
            jj = hi + j
            jj = jj[jj < len(t)]
            # tail only within contiguous bins
            jj = jj[idx[jj] - idx[hi[: len(jj)]] == j]
            bad[jj] = True
        bad |= ~np.isfinite(trend)
        idx, flux, sector, t = idx[~bad], flux[~bad], sector[~bad], t[~bad]
    # 3. detrend
    trend = flatten(t, flux, method="biweight", window_length=WINDOW_D, return_trend=True,
                    break_tolerance=GAP_D, edge_cutoff=0)[1]
    f = flux / trend
    ok = np.isfinite(f)
    idx, f, sector = idx[ok], f[ok], sector[ok]
    sig = _per_sector_sigma(f - 1, sector)
    ok = (f - 1) <= FLARE_SIG * sig
    idx, f, sector, sig = idx[ok], f[ok], sector[ok], sig[ok]
    sig = _per_sector_sigma(f - 1, sector)
    ok = np.isfinite(sig) & (sig > 0)
    return {"idx": idx[ok], "flux": f[ok], "sigma": sig[ok], "sector": sector[ok]}


# --------------------------------------------------------------------------------------------
# SES


@dataclass
class SES:
    k: int
    stride: int
    t: np.ndarray  # window centre (BTJD)
    N: np.ndarray
    D: np.ndarray
    sector: np.ndarray


def build_ses(pp: dict, k: int, stride: int, dense: bool = False):
    idx = pp["idx"]
    i0 = idx.min()
    L = int(idx.max() - i0 + 1)
    w = np.zeros(L)
    wd = np.zeros(L)
    ob = np.zeros(L)
    sec = np.zeros(L, np.int32)
    loc = idx - i0
    wi = 1.0 / pp["sigma"] ** 2
    w[loc] = wi
    wd[loc] = wi * (1.0 - pp["flux"])
    ob[loc] = 1.0
    sec[loc] = pp["sector"]
    sec = np.maximum.accumulate(sec)
    cw, cwd, cob = (np.concatenate([[0.0], np.cumsum(a)]) for a in (w, wd, ob))
    starts = np.arange(0, L - k + 1, stride)
    N = cwd[starts + k] - cwd[starts]
    D = cw[starts + k] - cw[starts]
    cov = cob[starts + k] - cob[starts]
    good = cov >= MIN_COVER * k
    all_starts = starts
    starts, N, D = starts[good], N[good], D[good]
    s = sec[starts + k // 2]
    ses = N / np.sqrt(D)
    for sv in np.unique(s):
        m = s == sv
        r = robust_sigma(ses[m])
        if np.isfinite(r) and r > 0:
            N[m] /= r**2
            D[m] /= r**2
    t = (i0 + starts + k / 2.0) * BIN_D
    if not dense:
        return SES(k, stride, t, N, D, s)
    n = len(all_starts)
    out = {"t_start": (i0 + k / 2.0) * BIN_D, "N": np.zeros(n), "D": np.zeros(n),
           "cnt": np.zeros(n), "s2": np.zeros(n)}
    j = starts // stride
    out["N"][j] = N
    out["D"][j] = D
    out["cnt"][j] = 1.0
    out["s2"][j] = np.where(N > 0, N * N / D, 0.0)
    return out


# --------------------------------------------------------------------------------------------
# fold


def fold_grid(t, N, D, periods, binw, dom_max):
    """Best valid MES per trial period, and its phase bin (bins of width binw from t=0)."""
    # +1e-6: samples sit on the bin lattice, and t/binw = i - eps would truncate into bin i-1
    u = np.ascontiguousarray(t / binw + 1e-6)
    s2 = np.where(N > 0, N * N / D, 0.0)
    return _fold(u, np.ascontiguousarray(N), np.ascontiguousarray(D), s2, periods / binw,
                 dom_max)


@njit(cache=True, fastmath=True)
def _fold(u, N, D, s2, pu, dom_max):
    n = len(u)
    nP = len(pu)
    best = np.zeros(nP)
    bestb = np.zeros(nP, np.int64)
    nbmax = int(pu.max()) + 2
    acc = np.zeros((nbmax, 4))  # sumN, sumD, count, max SES^2 (one cache line per bin)
    for j in range(nP):
        P = pu[j]
        inv = 1.0 / P
        nb = int(P) + 1
        acc[:nb, :] = 0.0
        for i in range(n):
            x = u[i]
            b = int(x - int(x * inv) * P)
            if b >= nb:
                b = nb - 1
            elif b < 0:
                b = 0
            acc[b, 0] += N[i]
            acc[b, 1] += D[i]
            acc[b, 2] += 1.0
            acc[b, 3] = max(acc[b, 3], s2[i])
        bm = 0.0
        bb = -1
        for b in range(nb):
            a0 = acc[b, 0]
            if acc[b, 2] >= 2.0 and a0 > 0:
                m2 = a0 * a0 / acc[b, 1]
                if acc[b, 3] <= dom_max * m2 and m2 > bm:
                    bm = m2
                    bb = b
        best[j] = math.sqrt(bm)
        bestb[j] = bb
    return best, bestb


@njit(cache=True, error_model="numpy")
def _ffa_best(X, p, m_real, dom_max):
    """Fast Folding Algorithm over an (m, p, 4) block: channels sumN, sumD, count (added)
    and max SES^2 (max-combined). Row r of output = fold with total drift r samples over
    (m - 1) row transitions, i.e. period p + r/(m-1) samples. Returns best MES and its bin per
    output row. Staelin (1969) recursion: out[s] = H[s//2] + roll(T[s//2], (s+1)//2)."""
    m = X.shape[0]
    A = X
    B = np.empty_like(X)
    n = 1
    while n < m:
        for s0 in range(0, m, 2 * n):
            for s in range(2 * n):
                h = s // 2
                off = ((s + 1) // 2) % p  # drift can exceed p when m > p
                hr = s0 + h
                tr = s0 + n + h
                o = s0 + s
                for x in range(p):
                    y = x + off
                    if y >= p:
                        y -= p
                    B[o, x, 0] = A[hr, x, 0] + A[tr, y, 0]
                    B[o, x, 1] = A[hr, x, 1] + A[tr, y, 1]
                    B[o, x, 2] = A[hr, x, 2] + A[tr, y, 2]
                    a3 = A[hr, x, 3]
                    b3 = A[tr, y, 3]
                    B[o, x, 3] = max(b3, a3)
        A, B = B, A
        n *= 2
    best = np.zeros(m)
    bestx = np.full(m, -1, np.int64)
    for r in range(m):
        bm = 0.0
        for x in range(p):
            a0 = A[r, x, 0]
            if A[r, x, 2] >= 2.0 and a0 > 0 and A[r, x, 1] > 0:
                m2 = a0 * a0 / A[r, x, 1]
                if A[r, x, 3] <= dom_max * m2 and m2 > bm:
                    bm = m2
                    bestx[r] = x
        best[r] = math.sqrt(bm)
    return best, bestx


def ffa_search(dense: dict, Plo: float, Phi: float, binw: float, dom_max: float):
    """Coarse search over all periods in [Plo, Phi] via FFA. dense: N, D, cnt, s2 on a regular
    grid of step binw starting at t_start (window centres). Returns (P, mes, t0) arrays."""
    L = len(dense["N"])
    base = np.stack([dense["N"], dense["D"], dense["cnt"], dense["s2"]], axis=1)
    Ps, Ms, T0s = [], [], []
    p_lo = max(2, int(math.floor(Plo / binw)))
    p_hi = int(math.ceil(Phi / binw))
    ops = 0
    for p in range(p_lo, p_hi):
        m_real = -(-L // p)
        if m_real < 2:
            break
        m = 1 << int(math.ceil(math.log2(m_real)))
        X = np.zeros((m * p, 4))
        X[:L] = base
        best, bx = _ffa_best(X.reshape(m, p, 4), p, m_real, dom_max)
        ops += m * p * int(math.log2(m))
        P = (p + np.arange(m) / (m - 1)) * binw
        ok = (P >= Plo) & (P <= Phi) & (bx >= 0)
        Ps.append(P[ok])
        Ms.append(best[ok])
        T0s.append(dense["t_start"] + bx[ok] * binw)
    if not Ps:
        return np.array([]), np.array([]), np.array([]), ops
    return np.concatenate(Ps), np.concatenate(Ms), np.concatenate(T0s), ops


def dense_ses(pp: dict, k: int, stride: int) -> dict:
    ses = build_ses(pp, k, stride, dense=True)
    return ses


def period_grid(lo: float, hi: float, binw: float, T: float) -> np.ndarray:
    r = 1.0 + binw / T
    n = int(math.log(hi / lo) / math.log(r)) + 1
    return lo * r ** np.arange(n + 1)


# --------------------------------------------------------------------------------------------
# driver


@dataclass
class Signal:
    P: float
    t0: float
    k: int
    mes: float
    depth: float
    n_ep: int
    dom: float
    epochs: list = field(default_factory=list)  # (t_mid, ses) for epochs with data

    def as_dict(self):
        return {"P": self.P, "t0": self.t0, "k": self.k, "dur_h": self.k * BIN_D * 24,
                "mes": self.mes, "depth": self.depth, "n_ep": self.n_ep, "dom": self.dom,
                "epochs": self.epochs}


def _epoch_breakdown(pp, k, P, t0):
    ses = build_ses(pp, k, 1)
    ph = (ses.t - t0 + 0.5 * P) % P - 0.5 * P
    m = np.abs(ph) < 0.5 * BIN_D + 1e-9
    ep = np.round((ses.t[m] - t0) / P).astype(int)
    N, D, tt = ses.N[m], ses.D[m], ses.t[m]
    out = {}
    for e, n_, d_, ti in zip(ep, N, D, tt):
        out.setdefault(e, [0.0, 0.0, ti])
        out[e][0] += n_
        out[e][1] += d_
    return out


def search_star(pp: dict, mass: float, radius: float, max_sig: int, floor: float = MES_FLOOR,
                coarse=None):
    """Iterative blind search. Returns (signals, stats).

    coarse: optional replacement for the coarse search, called as
    coarse(dense_ses, (Plo, Phi), binw, T) -> (P, mes, t0, ops); used for the GPU port."""
    pp = {k: v.copy() for k, v in pp.items()}
    T = (pp["idx"].max() - pp["idx"].min() + 1) * BIN_D
    plan = []
    for k in DURATIONS:
        pr = period_range_for(k, mass, radius)
        if pr is None:
            continue
        s = max(1, k // 2)  # amendment 2: coarse stride D/2 (pre-registered D/3)
        plan.append((k, s, pr))
    stats = {"T_base": T, "n_periods": 0, "ops": 0, "n_bins": len(pp["idx"]),
             "n_iter": 0}
    sigs: list[Signal] = []
    for _ in range(max_sig):
        best = (0.0, None)
        stats["n_iter"] += 1
        for k, s, pr in plan:
            dn = build_ses(pp, k, s, dense=True)
            if dn["cnt"].sum() < 3:
                continue
            binw = s * BIN_D
            p_sw = math.sqrt(FFA_ERR_MAX * T * binw)
            parts = []
            if coarse is not None:
                Pg, mg, tg, ops = coarse(dn, pr, binw, T)
                parts.append((Pg, mg, tg))
                stats["ops"] += ops
            elif pr[0] < p_sw:  # exact brute fold at short periods
                pg = period_grid(pr[0], min(pr[1], p_sw), binw, T)
                v = dn["cnt"] > 0
                tt = np.flatnonzero(v) * binw
                mb, bb = fold_grid(tt, dn["N"][v], dn["D"][v], pg, binw, DOMINANCE)
                ok = bb >= 0
                parts.append((pg[ok], mb[ok], dn["t_start"] + (bb[ok] + 0.5) * binw))
                stats["ops"] += int(v.sum()) * len(pg)
            if coarse is None and pr[1] > p_sw:
                Pf, mf, tf, ops = ffa_search(dn, max(pr[0], p_sw), pr[1], binw, DOMINANCE)
                parts.append((Pf, mf, tf))
                stats["ops"] += ops
            P = np.concatenate([a for a, _, _ in parts])
            mes = np.concatenate([b for _, b, _ in parts])
            t0s = np.concatenate([c for _, _, c in parts])
            o = np.argsort(P)
            P, mes, t0s = P[o], mes[o], t0s[o]
            if stats["n_iter"] == 1:
                stats["n_periods"] += len(P)
            if len(mes) == 0:
                continue
            j = int(np.argmax(mes))
            if mes[j] > best[0]:
                # local period step of the FFA grid around P[j]
                dP = float(np.median(np.abs(np.diff(P[max(j - 3, 0): j + 4])))) or s * BIN_D / T
                best = (float(mes[j]), (k, s, float(P[j]), dP, float(t0s[j])))
        if best[1] is None or best[0] < floor:
            break
        sig = refine(pp, plan, *best[1])
        if sig is None or sig.mes < floor:
            break
        sigs.append(sig)
        pp = mask(pp, sig.P, sig.t0, sig.k * BIN_D)
    return sigs, stats


def refine(pp, plan, k0, s0, P_c, dP, t0c):
    """Finer period/phase search around the coarse peak (+-3 coarse steps, 5x finer, stride 1),
    all allowed durations."""
    lo, hi = P_c - 3 * dP, P_c + 3 * dP
    best = None
    for k, s, pr in plan:
        if not (pr[0] <= P_c <= pr[1]):
            continue
        ses = build_ses(pp, k, 1)
        tref = ses.t.min()
        fine = np.linspace(lo, hi, 31)
        mes, bb = fold_grid(ses.t - tref, ses.N, ses.D, fine, BIN_D, DOMINANCE)
        i = int(np.argmax(mes))
        if best is None or mes[i] > best[0]:
            best = (float(mes[i]), k, float(fine[i]), tref + (bb[i] + 0.5) * BIN_D)
    if best is None or best[0] <= 0:
        return None
    mes, k, P, t0 = best
    eps = _epoch_breakdown(pp, k, P, t0)
    if len(eps) == 0:
        return None
    Ns = np.array([v[0] for v in eps.values()])
    Ds = np.array([v[1] for v in eps.values()])
    ses_i = Ns / np.sqrt(Ds)
    mes_e = Ns.sum() / math.sqrt(Ds.sum())
    dom = float((np.maximum(ses_i, 0) ** 2).max() / max(mes_e, 1e-9) ** 2) if mes_e > 0 else 1.0
    epochs = sorted((float(v[2]), float(v[0] / math.sqrt(v[1]))) for v in eps.values())
    return Signal(P=P, t0=t0, k=k, mes=mes, depth=float(Ns.sum() / Ds.sum()),
                  n_ep=len(eps), dom=dom, epochs=epochs)


def mask(pp, P, t0, dur):
    t = (pp["idx"] + 0.5) * BIN_D
    ph = (t - t0 + 0.5 * P) % P - 0.5 * P
    keep = np.abs(ph) > dur
    return {k: v[keep] for k, v in pp.items()}
