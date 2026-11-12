"""Vetting of one long-period detection (docs/spikes/mdwarf_vetting.md, pre-registered 2026-11-12).

Two steps, kept apart on purpose:
  measure(ctx, sig)        -> raw statistics of every test (no thresholds)
  grade(stats, thr, ...)   -> pass/soft/fail/n/a/error per test, strikes, tier and reasons
so thresholds can be frozen from DEV statistics and applied unchanged to TEST and the survey.

ctx comes from `prepare()` and holds the light curve exactly as the frozen gate-2 search saw it
(raw binned PDCSAP -> optional injection -> harm rotation removal -> gate-1 preprocessing ->
null transform), plus the rotation-removed raw flux, the SAP series and QUALITY flags.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "gate2"))
from common import BIN_D
from search import build_ses, central_duration, preprocess, robust_sigma

GAP_D = 0.5
ORBIT_D = 13.70
DUMP_BIT, STRAY_BITS = 32, 2048 | 4096
WIDTHS = (2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 24, 30, 36)
N_SCRAMBLE, N_ROTPER = 100, 100

# test -> applicable modes. "real" covers real stars and injections.
LC_TESTS = ("odd_even", "per_transit", "sector_consistency", "duration_density", "secondary",
            "model_compare", "uniqueness")
TIME_TESTS = ("sap", "systematics_window", "orbit_alias", "rotation_residual")
STAR_TESTS = ("pixel", "contamination", "catalogue")
ALL_TESTS = LC_TESTS + TIME_TESTS + STAR_TESTS


def applicable(test: str, mode: str) -> bool:
    if mode == "real":
        return True
    if mode == "inv":
        return test not in STAR_TESTS
    return test in LC_TESTS  # roll / invroll


# --------------------------------------------------------------------------------------------
# context


@dataclass
class Ctx:
    tic: int
    mass: float
    radius: float
    mode: str  # real | inv | roll | invroll
    pp: dict  # search input (after the null transform)
    raw: dict | None  # rotation-removed raw PDCSAP: idx, flux, sector, q (time tests)
    sap: dict | None  # preprocessed SAP (same pipeline) or None
    bkg: dict | None  # idx, bkg (fraction of the star's flux), sector
    crowd: dict = field(default_factory=dict)  # sector -> CROWDSAP
    prot: float | None = None  # median LS peak period of the harm census (0.1-5 d)
    rot_applied: bool = False
    star: dict | None = None  # catalogue / contamination info (catalog.py), None = missing
    pixel: dict | None = None  # pixel-check result (pixel_check.py), None = not run


def _inject(lc, inj, mass, radius, sap=None, crowd=None):
    from inject import binned_model

    out = dict(lc)
    out["flux"] = lc["flux"] * binned_model(lc["idx"], P=inj["P"], t0=inj["t0"], rp_re=inj["rp"],
                                            b=inj["b"], mass=mass, radius=radius)
    if sap is None:
        return out, None
    m = binned_model(sap["idx"], P=inj["P"], t0=inj["t0"], rp_re=inj["rp"], b=inj["b"],
                     mass=mass, radius=radius)
    c = np.array([crowd.get(int(s), 1.0) or 1.0 for s in sap["sector"]])
    s2 = dict(sap)
    s2["sap"] = sap["sap"] * (1.0 - c * (1.0 - m))  # a planet on the target, diluted in SAP
    return out, s2


def base_series(lc, sap_raw, crowd, inj=None, mass=None, radius=None):
    """Mode-independent preprocessing of one star (optionally injected)."""
    from search2 import remove_rotation

    if inj is not None:
        lc, sap_raw = _inject(lc, inj, mass, radius, sap_raw, crowd)
    lc2, infos = remove_rotation(lc)
    pp = preprocess(lc2)
    Ps = [v["P"] for v in infos.values() if v.get("P")]
    prot = float(np.median(Ps)) if Ps else None
    applied = any(v["applied"] for v in infos.values())
    raw = {"idx": lc["idx"], "flux": np.asarray(lc2["flux"], float), "sector": lc["sector"],
           "q": lc["q"]}
    sap_pp = bkg = None
    if sap_raw is not None and len(sap_raw["idx"]) > 100:
        sl = {"idx": sap_raw["idx"], "flux": np.asarray(sap_raw["sap"], float),
              "sector": sap_raw["sector"]}
        ok = np.isfinite(sl["flux"]) & (sl["flux"] > 0)
        sl = {k: v[ok] for k, v in sl.items()}
        try:
            s2, _ = remove_rotation(sl)
            sap_pp = preprocess(s2)
        except Exception:  # noqa: BLE001 - SAP unusable -> test error, not a crash
            sap_pp = None
        bkg = {"idx": sap_raw["idx"], "bkg": np.asarray(sap_raw["bkg"], float),
               "sector": sap_raw["sector"]}
    return {"pp": pp, "raw": raw, "sap": sap_pp, "bkg": bkg, "prot": prot, "applied": applied}


def make_ctx(base, tic, mass, radius, mode="real", seed=0, crowd=None, star=None):
    from run2 import _apply_mode

    pp = _apply_mode(base["pp"], mode, seed) if mode != "real" else base["pp"]
    raw, sap, bkg = base["raw"], base["sap"], base["bkg"]
    if mode == "inv":
        raw = dict(raw, flux=2.0 - raw["flux"])
        sap = dict(sap, flux=2.0 - sap["flux"]) if sap is not None else None
        if bkg is not None:
            bkg = dict(bkg, bkg=2 * np.nanmedian(bkg["bkg"]) - bkg["bkg"])
    elif mode in ("roll", "invroll"):
        raw = sap = bkg = None
    return Ctx(tic, mass, radius, mode, pp, raw, sap, bkg, crowd or {}, base["prot"],
               base["applied"], star)


# --------------------------------------------------------------------------------------------
# helpers


def _phase(t, P, t0):
    return (t - t0 + 0.5 * P) % P - 0.5 * P


def _epoch_windows(ses, P, t0):
    """Index of the stride-1 window centred on each epoch, and the epoch numbers."""
    ph = _phase(ses.t, P, t0)
    m = np.flatnonzero(np.abs(ph) < 0.5 * BIN_D + 1e-9)
    ep = np.round((ses.t[m] - t0) / P).astype(int)
    _, first = np.unique(ep, return_index=True)
    return m[first], ep[first]


def _segments_t(t, gap=GAP_D):
    br = np.flatnonzero(np.diff(t) > gap) + 1
    return np.split(np.arange(len(t)), br)


def _chi2_p(d, s):
    from scipy.stats import chi2

    w = 1 / s**2
    mu = np.sum(w * d) / np.sum(w)
    c2 = float(np.sum((d - mu) ** 2 * w))
    return c2, float(chi2.sf(c2, len(d) - 1)) if len(d) > 1 else 1.0


def _neglog(p):
    return float(-math.log10(max(p, 1e-300)))


def fold_phases(ses, P, t0, N=None, D=None, nbmin=2):
    """MES and dominance-valid MES at every 10-min phase bin of the fold at P."""
    N = ses.N if N is None else N
    D = ses.D if D is None else D
    nb = max(int(round(P / BIN_D)), 4)
    b = np.floor(((ses.t - t0) / P + 0.5 / nb) % 1.0 * nb).astype(int) % nb
    sN = np.bincount(b, N, nb)
    sD = np.bincount(b, D, nb)
    cnt = np.bincount(b, minlength=nb)
    s2 = np.zeros(nb)
    np.maximum.at(s2, b, np.where(N > 0, N * N / np.maximum(D, 1e-300), 0.0))
    s2n = np.zeros(nb)
    np.maximum.at(s2n, b, np.where(N < 0, N * N / np.maximum(D, 1e-300), 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        mes = np.where(sD > 0, sN / np.sqrt(sD), 0.0)
    ok = cnt >= nbmin
    dip_ok = ok & (mes > 0) & (s2 <= 0.8 * mes**2)
    bump_ok = ok & (mes < 0) & (s2n <= 0.8 * mes**2)
    phase = np.arange(nb) / nb * P
    phase = np.where(phase > P / 2, phase - P, phase)
    return phase, mes, dip_ok, bump_ok


# --------------------------------------------------------------------------------------------
# the tests (raw statistics)


def measure(ctx: Ctx, sig: dict) -> dict:
    P, t0, k = float(sig["P"]), float(sig["t0"]), int(sig["k"])
    Dd = k * BIN_D
    st: dict = {"P": P, "t0": t0, "k": k, "dur_h": Dd * 24, "mes": float(sig["mes"]),
                "mode": ctx.mode}
    pp = ctx.pp
    ses = build_ses(pp, k, 1)
    wi, ep = _epoch_windows(ses, P, t0)
    N, Dw = ses.N[wi], ses.D[wi]
    d_e, s_e = N / Dw, 1 / np.sqrt(Dw)
    ses_e = N / np.sqrt(Dw)
    mes_e = float(N.sum() / math.sqrt(Dw.sum())) if len(N) else 0.0
    st.update(n_transits=len(wi), mes_vet=mes_e, depth=float(N.sum() / Dw.sum()) if len(N)
              else float("nan"), epochs=[float(x) for x in ses.t[wi]],
              epoch_ses=[float(x) for x in ses_e], epoch_n=[int(x) for x in ep],
              epoch_sector=[int(x) for x in ses.sector[wi]])
    if len(wi) == 0:
        st["error"] = "no epoch windows"
        return st

    # 2 odd/even
    odd = ep % 2 == 1
    if odd.any() and (~odd).any():
        do, so = N[odd].sum() / Dw[odd].sum(), 1 / math.sqrt(Dw[odd].sum())
        de, se = N[~odd].sum() / Dw[~odd].sum(), 1 / math.sqrt(Dw[~odd].sum())
        st["odd_even"] = {"z": float(abs(do - de) / math.hypot(so, se)), "d_odd": float(do),
                          "d_even": float(de)}
    else:
        st["odd_even"] = None

    # 3 per-transit
    c2, p = _chi2_p(d_e, s_e)
    st["per_transit"] = {"dom": float(np.max(np.maximum(ses_e, 0) ** 2) / max(mes_e, 1e-9) ** 2),
                         "neg_logp": _neglog(p), "minses_neg": float(-ses_e.min()),
                         "chi2": c2}

    # 4 sector consistency (+ duration halves, info)
    secs = ses.sector[wi]
    us = np.unique(secs)
    if len(us) >= 2:
        dd = np.array([N[secs == s].sum() / Dw[secs == s].sum() for s in us])
        ss = np.array([1 / math.sqrt(Dw[secs == s].sum()) for s in us])
        _, p = _chi2_p(dd, ss)
        st["sector_consistency"] = {"neg_logp": _neglog(p), "n_sec": len(us)}
    else:
        st["sector_consistency"] = None

    # 5 duration vs density (width scan at fixed P), and per-half widths (info)
    centres = t0 + ep * P
    best = (-1.0, k)
    half = {"early": (-1.0, k), "late": (-1.0, k)}
    order = np.argsort(centres)
    early = np.zeros(len(centres), bool)
    early[order[: max(1, len(centres) // 2)]] = True
    for kk in WIDTHS:
        s2_ = build_ses(pp, kk, 1)
        for off in np.arange(-k // 2, k // 2 + 1):
            tt = centres + off * BIN_D
            j = np.searchsorted(s2_.t, tt)
            j = np.clip(j, 0, len(s2_.t) - 1)
            hit = np.abs(s2_.t[j] - tt) < 0.5 * BIN_D + 1e-9
            if not hit.any():
                continue
            nn, dd_ = s2_.N[j[hit]], s2_.D[j[hit]]
            m_ = nn.sum() / math.sqrt(dd_.sum())
            if m_ > best[0]:
                best = (m_, kk)
            for name, sel in (("early", early[hit]), ("late", ~early[hit])):
                if sel.any():
                    mh = nn[sel].sum() / math.sqrt(dd_[sel].sum())
                    if mh > half[name][0]:
                        half[name] = (mh, kk)
    dc = float(central_duration(P, ctx.mass, ctx.radius))
    st["duration_density"] = {"ratio": float(best[1] * BIN_D / dc), "k_fit": int(best[1]),
                              "dc_h": dc * 24}
    st["duration_halves_info"] = {"k_early": int(half["early"][1]), "k_late": int(half["late"][1])}

    # 9 + 11 fold over phases: secondary and uniqueness
    phase, fmes, dip_ok, bump_ok = fold_phases(ses, P, t0, nbmin=min(2, len(wi)))
    excl = np.abs(phase) < max(2 * Dd, 0.03 * P)
    oth = ~excl & dip_ok
    vals = fmes[~excl]
    med, rs = float(np.median(vals)), float(robust_sigma(vals))
    mes_sec = float(fmes[oth].max()) if oth.any() else 0.0
    E = math.sqrt(2 * math.log(max(P / Dd, 2.0)))
    zsec = (mes_sec - med) / rs if rs > 0 else 0.0
    st["secondary"] = {"excess": float(zsec - E), "mes_sec": mes_sec, "z_sec": float(zsec),
                       "E": E, "phase_sec": float(phase[oth][np.argmax(fmes[oth])]) if oth.any()
                       else None}
    mes_pos = float(-fmes[bump_ok].min()) if bump_ok.any() else 0.0
    margin = st["mes"] - max(mes_pos, mes_sec)
    rng = np.random.default_rng(int(ctx.tic) % 2**31 + int(round(P * 1000)))
    segs = _segments_t(ses.t)
    hits = 0
    for _ in range(N_SCRAMBLE):
        Ns, Ds = ses.N.copy(), ses.D.copy()
        for sg in segs:
            if len(sg) > 1:
                sh = int(rng.integers(0, len(sg)))
                Ns[sg] = np.roll(Ns[sg], sh)
                Ds[sg] = np.roll(Ds[sg], sh)
        _, m2, ok2, _ = fold_phases(ses, P, t0, Ns, Ds, nbmin=min(2, len(wi)))
        if ok2.any() and m2[ok2].max() >= st["mes"]:
            hits += 1
    st["uniqueness"] = {"neg_margin": float(-margin), "p_scr": hits / N_SCRAMBLE,
                        "mes_pos": mes_pos, "mes_2nd": mes_sec}

    # 10 model comparison
    st["model_compare"] = _model_compare(pp, P, t0, Dd, centres, ctx.prot)

    # time-anchored tests
    st["sap"] = _sap_test(ctx, k, P, t0, centres, st["depth"]) if ctx.raw is not None else None
    if ctx.raw is not None:
        st["systematics_window"] = _sys_window(ctx.raw, centres, Dd, ses_e)
        st["orbit_alias"] = _orbit_alias(ctx.raw, centres, P)
        st["rotation_residual"] = _rot_residual(ctx, P, t0, Dd)
    else:
        st["systematics_window"] = st["orbit_alias"] = st["rotation_residual"] = None
    return st


def _wls(cols, y, w):
    A = np.column_stack(cols) * np.sqrt(w)[:, None]
    b = y * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ coef
    return float(r @ r)


def _model_compare(pp, P, t0, Dd, centres, prot):
    t = (pp["idx"] + 0.5) * BIN_D
    W = max(3 * Dd, 0.3)
    y = pp["flux"] - 1.0
    w = 1 / pp["sigma"] ** 2
    sel, eid, dt = [], [], []
    for i, c in enumerate(centres):
        j = np.flatnonzero(np.abs(t - c) < W)
        if len(j) < 5:
            continue
        sel.append(j)
        eid.append(np.full(len(j), i))
        dt.append(t[j] - c)
    if not sel:
        return None
    j = np.concatenate(sel)
    eid = np.concatenate(eid)
    dt = np.concatenate(dt)
    y, w, tt = y[j], w[j], t[j]
    inn = np.abs(dt) < Dd / 2
    # DEV fix (2026-11-12): lone extreme points outside the transit (up to 1e11 in chi^2 on
    # some stars) dominated every fit; out-of-transit points are winsorised at 8 sigma.
    s = 1 / np.sqrt(w)
    y = np.where(inn, y, np.clip(y, -8 * s, 8 * s))
    ue = np.unique(eid)
    off = [(eid == e).astype(float) for e in ue]
    n = len(y)
    ln = math.log(n)
    box = [-inn.astype(float)]
    # DEV fix (2026-11-12): NESTED comparison. For each alternative X the statistic is how much
    # the box still improves the fit once X is in: dBIC_X = BIC(X) - BIC(X + box). The first
    # version compared X and box separately, so a per-epoch sinusoid absorbing unrelated
    # rotation residue "beat" a real transit by 1e5 on a fast rotator.
    alts = {}
    # step: per-epoch amplitude, location common among {-D/2, 0, +D/2} (best for the step)
    best = None
    for loc in (-Dd / 2, 0.0, Dd / 2):
        cols = off + [((eid == e) & (dt > loc)).astype(float) for e in ue]
        c2 = _wls(cols, y, w)
        if best is None or c2 < best[0]:
            best = (c2, cols)
    alts["step"] = best[1]
    # sinusoid at P_rot/h, phase free per epoch
    best = None
    if prot:
        for h in (1, 2, 3):
            Ph = prot / h
            if Ph < 2 * Dd:
                continue
            cols = list(off)
            for e in ue:
                m = (eid == e).astype(float)
                cols += [m * np.cos(2 * np.pi * tt / Ph), m * np.sin(2 * np.pi * tt / Ph)]
            c2 = _wls(cols, y, w)
            if best is None or c2 < best[0]:
                best = (c2, cols)
    if best is not None:
        alts["sine"] = best[1]
    # single outlier: the most negative in-transit point vs per-epoch offsets
    pen = {}
    if inn.any():
        mu = np.array([np.sum(w[eid == e] * y[eid == e]) / np.sum(w[eid == e]) for e in ue])
        res = y - mu[np.searchsorted(ue, eid)]
        cand = np.flatnonzero(inn)
        jj = cand[np.argmin(res[cand] * np.sqrt(w[cand]))]
        col = np.zeros(n)
        col[jj] = 1.0
        alts["outlier"] = off + [col]
    d = {}
    for k_, cols in alts.items():
        d[k_] = _wls(cols, y, w) - _wls(cols + box, y, w) - ln - pen.get(k_, 0.0)
    c_null = _wls(off, y, w)
    c_box = _wls(off + box, y, w)
    kbest = min(d, key=d.get)
    return {"neg_dbic": float(-d[kbest]), "dbic": float(d[kbest]), "best_alt": kbest,
            "dbic_null": float(c_null - c_box - ln),
            "dbic_each": {k_: float(v) for k_, v in d.items()}}


def _sap_test(ctx, k, P, t0, centres, d_pdc):
    if ctx.sap is None:
        return {"error": "no SAP"}
    try:
        ses = build_ses(ctx.sap, k, 1)
    except Exception as e:  # noqa: BLE001
        return {"error": f"SAP SES: {e!r}"[:100]}
    wi, ep = _epoch_windows(ses, P, t0)
    if len(wi) == 0:
        return {"error": "no SAP data at the epochs"}
    N, D = ses.N[wi], ses.D[wi]
    d_sap, s_sap = N.sum() / D.sum(), 1 / math.sqrt(D.sum())
    c = np.array([ctx.crowd.get(int(s), 1.0) or 1.0 for s in ses.sector[wi]])
    cbar = float(np.sum(c * D) / D.sum())
    z = (d_sap - cbar * d_pdc) / s_sap
    out = {"z_neg": float(-z), "d_sap": float(d_sap), "d_exp": float(cbar * d_pdc),
           "crowd": cbar, "mes_sap": float(N.sum() / math.sqrt(D.sum()))}
    # background rise during transit
    if ctx.bkg is not None:
        tb = (ctx.bkg["idx"] + 0.5) * BIN_D
        b = ctx.bkg["bkg"]
        dl, wl = [], []
        Dd = k * BIN_D
        for c0 in centres:
            near = np.abs(tb - c0)
            inn = near < Dd / 2
            loc = (near > Dd / 2 + 1 / 24) & (near < Dd / 2 + 0.5)
            if inn.sum() < 1 or loc.sum() < 6:
                continue
            bl = b[loc]
            sig = robust_sigma(bl)
            if not (np.isfinite(sig) and sig > 0):
                continue
            dl.append(np.mean(b[inn]) - np.median(bl))
            wl.append(inn.sum() / sig**2)
        if dl:
            dl, wl = np.array(dl), np.array(wl)
            rise = float(np.sum(dl * wl) / wl.sum())
            out["bkg_z"] = float(rise * math.sqrt(wl.sum()))
            out["bkg_rise_frac"] = float(rise / max(cbar * d_pdc, 1e-9))
            # the graded statistic: z only counts when the rise is > 50% of the dip
            out["bkg_stat"] = out["bkg_z"] if out["bkg_rise_frac"] > 0.5 else 0.0
    return out


def _sys_window(raw, centres, Dd, ses_e):
    t = (raw["idx"] + 0.5) * BIN_D
    q = raw["q"]
    segs = _segments_t(t)
    starts = np.array([t[s[0]] for s in segs])
    ends = np.array([t[s[-1]] for s in segs])
    flags = []
    for c in centres:
        why = []
        if np.min(np.abs(np.concatenate([starts - c, ends - c]))) < 0.5 + Dd / 2:
            why.append("edge")
        w = np.abs(t - c) < Dd / 2 + 1 / 24
        if np.any(q[w] & DUMP_BIT):
            why.append("dump")
        inn = np.abs(t - c) < Dd / 2
        if inn.any() and np.mean((q[inn] & STRAY_BITS) > 0) >= 0.5:
            why.append("scattered")
        flags.append(why)
    s2 = np.maximum(np.asarray(ses_e), 0) ** 2
    fl = np.array([bool(f) for f in flags])
    share = float(s2[fl].sum() / s2.sum()) if s2.sum() > 0 else 0.0
    return {"share": share, "flags": flags}


def _orbit_alias(raw, centres, P):
    t = (raw["idx"] + 0.5) * BIN_D
    segs = _segments_t(t)
    starts = np.array([t[s[0]] for s in segs])
    since = []
    for c in centres:
        j = np.searchsorted(starts, c) - 1
        if j >= 0:
            since.append(c - starts[j])
    spread = float(np.ptp(since)) if len(since) >= 2 else float("inf")
    alias = [n for n in (1, 2, 3, 4) if abs(P / (n * ORBIT_D) - 1) < 0.02]
    return {"spread": spread, "alias_n": alias[0] if alias else None,
            "since_start": [float(x) for x in since]}


def _rot_residual(ctx, P, t0, Dd):
    raw = ctx.raw
    t = (raw["idx"] + 0.5) * BIN_D
    y = raw["flux"].astype(float) - 1.0
    ok = np.isfinite(y) & (np.abs(_phase(t, P, t0)) > Dd)
    t, y = t[ok], y[ok]
    for sg in _segments_t(t):
        y[sg] -= np.median(y[sg])
    s = robust_sigma(y)
    keep = y < 3 * s
    t, y = t[keep], y[keep]
    if len(t) < 200:
        return {"error": "too little raw data"}

    def power(per):
        tot = 0.0
        for pp_ in (per, per / 2):
            c, sn = np.cos(2 * np.pi * t / pp_), np.sin(2 * np.pi * t / pp_)
            A = np.column_stack([c, sn])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            tot += float(np.sum((A @ coef) ** 2))
        return tot

    p0 = power(P)
    rng = np.random.default_rng(int(ctx.tic) % 2**31 + 7)
    rp = np.exp(rng.uniform(math.log(10), math.log(60), N_ROTPER))
    rp = rp[np.abs(rp / P - 1) > 0.05]
    pr = np.array([power(x) for x in rp])
    p_rot = float((1 + np.sum(pr >= p0)) / (1 + len(pr)))
    return {"neg_logp": _neglog(p_rot), "p_rot": p_rot}


def attach_star(stats: dict, info: dict | None) -> dict:
    """Star-level tests for this signal: contamination (needs the depth) and the catalogue."""
    if info is None:
        stats["star"] = None
        return stats
    g = info.get("gaia") or {"error": "missing"}
    if g.get("error"):
        cont = {"error": g["error"]}
    else:
        depth = max(float(stats.get("depth") or 0.0), 1e-6)
        nb1 = [n for n in g["neighbours"] if n["sep"] <= 21.0]
        needs = [depth / 10 ** (-0.4 * max(n["dG"], -5)) for n in nb1]
        cont = {"min_need": float(min(needs)) if needs else None, "n_nb_1px": len(nb1),
                "n_nb_2px": sum(n["sep"] <= 42.0 for n in g["neighbours"]),
                "brightest_1px_dG": float(min(n["dG"] for n in nb1)) if nb1 else None}
    stats["star"] = {"contamination": cont, "catalogue": info.get("catalogue")}
    return stats


# --------------------------------------------------------------------------------------------
# grading


# statistic -> (path in stats, default soft, default fail); larger = worse. None = no hard fail.
DEFAULT_THR = {
    "odd_even.z": (3.0, 4.0),
    "per_transit.dom": (0.65, None),
    "per_transit.neg_logp": (2.0, 3.0),
    "per_transit.minses_neg": (1.0, 2.0),
    "sector_consistency.neg_logp": (2.0, 3.0),
    "duration_density.ratio": (1.3, 2.0),
    "secondary.excess": (1.0, 2.0),
    "model_compare.neg_dbic": (-6.0, 0.0),
    "uniqueness.neg_margin": (-2.0, -1.0),
    "uniqueness.p_scr": (0.01, 0.05),
    "sap.z_neg": (3.0, 4.0),
    "sap.bkg_stat": (4.0, None),
    "systematics_window.share": (0.5, 0.8),
    "orbit_alias.neg_spread": (-0.5, None),  # fail handled with the alias rule below
    "rotation_residual.neg_logp": (1.69897, None),  # p_rot < 0.02
    "contamination.need_neg": (-0.5, None),  # soft if a neighbour needs <= 50% eclipse
}


def stat_value(stats, key):
    test, name = key.split(".")
    d = stats.get(test)
    if test == "orbit_alias" and d:
        return -d["spread"] if np.isfinite(d["spread"]) else None
    if test == "contamination":
        d = (stats.get("star") or {}).get("contamination")
        if not d or d.get("error"):
            return None
        return -d["min_need"] if d.get("min_need") is not None else None
    if not d or d.get("error"):
        return None
    v = d.get(name)
    return None if v is None or not np.isfinite(v) else float(v)


def _level(v, soft, fail):
    if v is None:
        return None
    if fail is not None and v > fail:
        return "fail"
    if v > soft:
        return "soft"
    return "pass"


def grade(stats: dict, thr: dict | None = None, blind: bool = False) -> dict:
    """Per-test status, strikes, tier (A/B/C/K) and reasons. blind=True switches the catalogue
    test off (hold-out evaluation)."""
    thr = thr or DEFAULT_THR
    mode = stats.get("mode", "real")
    res, reasons = {}, []
    order = {"pass": 0, "soft": 1, "fail": 2}

    def put(test, status, why=None):
        old = res.get(test)
        if old in order and status in order and order[old] >= order[status]:
            return
        res[test] = status
        if why and status in ("soft", "fail"):
            reasons.append(f"{test}: {why}")

    for key, (soft, fail) in thr.items():
        test = key.split(".")[0]
        if not applicable(test, mode):
            res[test] = "n/a"
            continue
        if test == "contamination":
            continue  # graded with the star tests below
        d = stats.get(test)
        if d is None:
            res.setdefault(test, "n/a")
            continue
        if d.get("error"):
            res[test] = "error"
            reasons.append(f"{test}: could not be computed ({d['error']})")
            continue
        v = stat_value(stats, key)
        lv = _level(v, soft, fail)
        if lv is None:
            res.setdefault(test, "n/a")
            continue
        if test == "orbit_alias" and lv == "soft" and d.get("alias_n"):
            lv = "fail"
        put(test, lv, f"{key.split('.')[1]} = {v:.3g} (soft > {soft:g}"
                      + (f", fail > {fail:g})" if fail is not None else ")"))
    # star-level tests
    star = stats.get("star")
    known = False
    if mode == "real":
        if star is None:
            for t_ in ("contamination", "catalogue"):
                res[t_] = "error"
                reasons.append(f"{t_}: catalogue information missing")
        else:
            c = star.get("contamination") or {"error": "missing"}
            if c.get("error"):
                res["contamination"] = "error"
                reasons.append(f"contamination: {c['error']}")
            else:
                soft = thr["contamination.need_neg"][0]
                v = -c["min_need"] if c.get("min_need") is not None else None
                lv = _level(v, soft, None) if v is not None else "pass"
                put("contamination", lv, f"a Gaia neighbour could make the dip with a "
                    f"{100 * (c.get('min_need') or 1):.0f}% eclipse")
            if blind:
                res["catalogue"] = "n/a"
            else:
                cat = _catalogue_status(star, stats["P"])
                res["catalogue"] = cat["status"]
                known = cat["known"]
                if cat["status"] in ("soft", "fail", "error"):
                    reasons.append(f"catalogue: {cat['why']}")
        px = stats.get("pixel")
        if px is None:
            res["pixel"] = "n/a"
        elif px.get("error"):
            res["pixel"] = "error"
            reasons.append(f"pixel: {px['error']}")
        else:
            res["pixel"] = px["status"]
            if px["status"] in ("soft", "fail"):
                reasons.append(f"pixel: {px['why']}")
    for t_ in ALL_TESTS:
        res.setdefault(t_, "n/a")
    n_tr = int(stats.get("n_transits", 0))
    n_soft = sum(v == "soft" for v in res.values())
    hard = [k for k, v in res.items() if v == "fail"]
    strikes = n_soft + (1 if n_tr == 2 else 0)
    errors = [k for k, v in res.items() if v == "error" and k != "pixel"]
    if stats.get("error"):
        tier = "C"
        reasons.append(stats["error"])
    elif known:
        tier = "K"
    elif hard or strikes >= 2:
        tier = "C"
    elif strikes == 0 and n_tr >= 3 and not errors:
        tier = "A"
    else:
        tier = "B"
    if n_tr == 2:
        reasons.append("only 2 transits: the period could be P/2, P/3, ... (strike)")
    return {"tier": tier, "status": res, "strikes": strikes, "hard": hard, "errors": errors,
            "reasons": reasons}


def _catalogue_status(star, P):
    cat = star.get("catalogue") or {}
    if cat.get("error"):
        return {"status": "error", "known": False, "why": cat["error"]}
    errs = [k for k, v in cat.get("sources", {}).items() if v != "ok"]

    def alias(p2):
        r = P / p2
        for x in (r, 1 / r):
            n = round(x)
            if 1 <= n <= 6 and abs(x - n) <= 0.01 * n:
                return True
        return False

    fps = [o for o in cat.get("objects", []) if o["kind"] == "toi_fp" and o.get("P")
           and alias(o["P"])]
    if fps:
        return {"status": "fail", "known": False,
                "why": f"{fps[0]['name']} at this period is dispositioned a false positive"}
    known = [o for o in cat.get("objects", []) if o["kind"] in ("planet", "toi", "ctoi")
             and o.get("P") and alias(o["P"])]
    if known:
        o = known[0]
        return {"status": "info", "known": True, "why": f"matches {o['name']} (P = {o['P']:.4f} d)"}
    ebs = [o for o in cat.get("objects", []) if o["kind"] == "eb"]
    eb_p = [o for o in ebs if o.get("P") and alias(o["P"])]
    if eb_p:
        return {"status": "fail", "known": False,
                "why": f"known eclipsing binary {eb_p[0]['name']} at a matching period"}
    if errs:
        return {"status": "error", "known": False,
                "why": "query failed for " + ", ".join(errs) + " (NOT the same as 'not known')"}
    if ebs:
        return {"status": "soft", "known": False,
                "why": f"catalogued eclipsing binary within 21\" ({ebs[0]['name']}), other period"}
    return {"status": "pass", "known": False, "why": "no known planet, TOI, CTOI or EB"}
