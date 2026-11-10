"""Render one PNG card per gallery candidate.

Layout (BTJD on every x axis):

1. The whole sector: PDCSAP (black) and SAP (orange, offset below), momentum dumps (red
   ticks), scattered-light bins (orange band), other quality-flagged bins (grey band), the
   zoom window (blue band) and the time-resolved anomaly (red fill, right axis).
2. Left: zoom on the most anomalous window. Right: the same window for the candidate (top,
   black) and its nearest CCD neighbours, each on its own y scale. A dip at the same time in
   neighbours several pixels away is a systematic.
3. The star in its other ingested sectors.

Text context (scores, catalogue matches) lives in the HTML page, not in the PNG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from oddball.score.classical import ANOM_SNR, box_search, centred_box_snr, channel_features
from oddball.tess.plots import mark_systematics, overlay_surprise, robust_ylim

BIN_DAYS = 30.0 / 1440.0
TESS_PIXEL_ARCSEC = 21.0


@dataclass
class Window:
    t0: float
    t1: float
    channel: str  # "pdc" or "sap": the channel the neighbours are shown in
    source: str  # human-readable: how the window was found


@dataclass
class Trace:
    """A time-resolved anomaly to overlay: values on times ``t``."""

    t: np.ndarray
    v: np.ndarray
    label: str


@dataclass
class Neighbour:
    tic: int
    tmag: float
    sep_arcsec: float
    lc: dict | None


@dataclass
class OtherSector:
    sector: int
    camera: int
    ccd: int
    lc: dict


@dataclass
class CardData:
    tic: int
    sector: int
    title: str
    lc: dict
    dump_times: np.ndarray
    window: Window
    trace: Trace | None
    neighbours: list[Neighbour] = field(default_factory=list)
    others: list[OtherSector] = field(default_factory=list)


def norm(f: np.ndarray) -> np.ndarray:
    """Relative flux ``f / median - 1`` (NaN-safe; all-NaN if the median is not positive)."""
    f = np.asarray(f, np.float64)
    med = np.nanmedian(f) if np.isfinite(f).any() else np.nan
    if not np.isfinite(med) or med <= 0:
        return np.full_like(f, np.nan)
    return f / med - 1.0


def sector_dump_times(lcs, min_frac: float = 0.3) -> np.ndarray:
    """BTJD of bins flagged as momentum dumps (QUALITY bit 32) in >= ``min_frac`` of the light
    curves that cover them. Dumps are spacecraft-wide, so a few hundred stars pin them down."""
    flagged: dict[int, int] = {}
    cover: dict[int, int] = {}
    for lc in lcs:
        g = int(lc["bin0"]) + np.arange(len(lc["qflags"]))
        q = np.asarray(lc["qflags"]).astype(np.int64)
        for b in g:
            cover[b] = cover.get(b, 0) + 1
        for b in g[(q & 32) != 0]:
            flagged[b] = flagged.get(b, 0) + 1
    bins = sorted(b for b, k in flagged.items() if k >= min_frac * cover[b])
    return (np.asarray(bins, np.float64) + 0.5) * BIN_DAYS


def segment_edges(lc: dict, min_gap_days: float = 0.75) -> np.ndarray:
    """BTJD of the start and end of each stretch of data (orbit edges, downlink gaps)."""
    t = np.asarray(lc["time"], np.float64)
    ok = np.isfinite(np.asarray(lc["flux"], np.float64)) | np.isfinite(
        np.asarray(lc["sap_flux"], np.float64))  # fmt: skip
    t = t[ok]
    if not len(t):
        return np.array([])
    gap = np.flatnonzero(np.diff(t) > min_gap_days)
    return np.unique(np.r_[t[0], t[gap], t[gap + 1], t[-1]])


def _clip_halfwidth(hw: float) -> float:
    return float(np.clip(hw, 0.75, 3.0))


def window_from_trace(tr: Trace) -> Window:
    """Centre on the peak of the time-resolved anomaly (smoothed over 8 h)."""
    v = np.nan_to_num(np.asarray(tr.v, np.float64), nan=0.0)
    k = 16
    sm = np.convolve(v, np.ones(k) / k, mode="same")
    i = int(np.argmax(sm))
    above = sm >= 0.5 * sm[i]
    lo = i
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = i
    while hi < len(sm) - 1 and above[hi + 1]:
        hi += 1
    hw = _clip_halfwidth(0.75 * (tr.t[hi] - tr.t[lo]) + 0.5)
    c = float(tr.t[i])
    return Window(c - hw, c + hw, "pdc", f"peak of {tr.label} at BTJD {c:.2f}")


def classical_window(lc: dict) -> tuple[Window, Trace]:
    """Fallback: the most significant box dip or bump in PDCSAP or SAP (the classical box
    search, ``oddball.score.classical``), and the 2-h dip SNR of that channel as the trace."""
    t = np.asarray(lc["time"], np.float64)
    best = None
    traces = {}
    for ch, key in (("pdc", "flux"), ("sap", "sap_flux")):
        f = np.asarray(lc[key], np.float64)[None, :]
        err = np.asarray(lc["flux_err"], np.float64)[None, :]
        if not np.isfinite(f).any():
            continue
        out = channel_features(f, err, t, BIN_DAYS)
        sig = np.nan_to_num(out.feats["sigma_w"], nan=1.0)
        traces[ch] = centred_box_snr(out.resid, sig)[0]
        for sign, kind in ((-1, "dip"), (1, "brightening")):
            snr, start, width = box_search(out.resid, sig, sign)
            if np.isfinite(snr[0]) and (best is None or snr[0] > best[0]):
                best = (float(snr[0]), ch, kind, int(start[0]), int(width[0]))
    if best is None:
        c = float(np.nanmedian(t))
        return Window(c - 1, c + 1, "pdc", "no valid data"), None
    snr, ch, kind, start, width = best
    c = float(t[min(start + width // 2, len(t) - 1)])
    hw = _clip_halfwidth(1.5 * width * BIN_DAYS)
    name = {"pdc": "PDCSAP", "sap": "SAP"}[ch]
    w = Window(c - hw, c + hw, ch, f"classical box search: strongest {name} {kind} "
               f"(SNR {snr:.0f}, {width * BIN_DAYS * 24:.0f} h box) at BTJD {c:.2f}")  # fmt: skip
    s = traces[ch]
    tr = Trace(t, np.where(s >= ANOM_SNR, s, 0.0), f"classical 2-h {name} dip SNR (>= {ANOM_SNR:g})")
    return w, tr


def _span(*arrays) -> tuple[float, float] | None:
    v = np.concatenate([np.asarray(a, np.float64).ravel() for a in arrays])
    v = v[np.isfinite(v)]
    return (float(v.min()), float(v.max())) if len(v) else None


def _plot_lc(ax, lc: dict, ms: float = 1.5, sap_offset: bool = True,
             window: Window | None = None) -> None:  # fmt: skip
    """PDCSAP and SAP (offset below). Y limits are robust (0.5-99.5 percentile), widened to
    show every point inside ``window`` in full, since that is where the anomaly is."""
    t = lc["time"]
    pdc = norm(lc["flux"])
    sap = norm(lc["sap_flux"])
    inw = (t >= window.t0) & (t <= window.t1) if window is not None else np.zeros(len(t), bool)
    off = 0.0
    if sap_offset:
        r = 0.01
        if np.isfinite(pdc).any():
            r = np.nanpercentile(pdc, 99.5) - np.nanpercentile(pdc, 0.5)
            sw = _span(pdc[inw], sap[inw])
            if sw is not None:
                r = max(r, sw[1] - sw[0])
        off = -max(1.3 * r, 0.003)
    ax.plot(t, sap + off, ".", ms=ms * 0.8, color="tab:orange", alpha=0.7,
            label="SAP" + (" (offset)" if sap_offset else ""))  # fmt: skip
    ax.plot(t, pdc, ".", ms=ms, color="k", label="PDCSAP")
    robust_ylim(ax, pdc, sap + off)
    sw = _span(pdc[inw], sap[inw] + off)
    if sw is not None:
        lo, hi = ax.get_ylim()
        pad = 0.05 * (max(hi, sw[1]) - min(lo, sw[0]))
        ax.set_ylim(min(lo, sw[0] - pad), max(hi, sw[1] + pad))


def _in(lc: dict, w: Window, pad: float = 0.0):
    t = lc["time"]
    return (t >= w.t0 - pad) & (t <= w.t1 + pad)


def render(card: CardData, path: Path, dpi: int = 80) -> None:
    n_nb = len(card.neighbours)
    n_other = max(1, len(card.others))
    h3 = 0.95 if card.others else 0.12
    fig = plt.figure(figsize=(14, 9.6 + 0.9 * h3))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.6, h3], width_ratios=[1.25, 1.0],
                          hspace=0.42, wspace=0.12, left=0.05, right=0.985, top=0.95,
                          bottom=0.05)  # fmt: skip
    lc, w = card.lc, card.window

    # 1. full sector
    ax = fig.add_subplot(gs[0, :])
    mark_systematics(ax, lc["time"], lc["qflags"], card.dump_times, other_flags=True)
    ax.axvspan(w.t0, w.t1, color="tab:blue", alpha=0.08, lw=0, zorder=0)
    edges = segment_edges(lc)
    for e in edges:
        ax.axvline(e, color="tab:purple", lw=0.8, ls=":", zorder=0)
    _plot_lc(ax, lc, window=w)
    if card.trace is not None:
        overlay_surprise(ax, card.trace.t, card.trace.v)
    ax.set_title(card.title, fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="upper right", markerscale=4, ncol=2)
    tr_label = card.trace.label if card.trace is not None else "none"
    ax.set_xlabel(f"BTJD   red lines = momentum dumps; purple dotted = start/end of an orbit "
                  f"(data gap); orange band = scattered light; grey band = other quality flags; "
                  f"blue band = zoom window; red fill = {tr_label}", fontsize=7.5)  # fmt: skip
    ax.tick_params(labelsize=7)
    ax.set_ylabel("relative flux", fontsize=8)

    # 2a. zoom
    az = fig.add_subplot(gs[1, 0])
    m = _in(lc, w)
    sub = {k: (v[m] if isinstance(v, np.ndarray) and v.shape == lc["time"].shape else v)
           for k, v in lc.items()}  # fmt: skip
    # zoom normalises by the whole-sector median so depths match the top panel
    full_pdc, full_sap = norm(lc["flux"])[m], norm(lc["sap_flux"])[m]
    dumps_in = card.dump_times[(card.dump_times >= w.t0) & (card.dump_times <= w.t1)]
    mark_systematics(az, sub["time"], sub["qflags"], dumps_in, other_flags=True)
    for e in edges[(edges >= w.t0) & (edges <= w.t1)]:
        az.axvline(e, color="tab:purple", lw=0.8, ls=":", zorder=0)
    r = np.nanpercentile(full_pdc, 99) - np.nanpercentile(full_pdc, 1) if np.isfinite(
        full_pdc).any() else 0.01  # fmt: skip
    off = -max(1.2 * r, 0.002)
    az.errorbar(sub["time"], full_pdc, yerr=sub["flux_err"], fmt=".", ms=3, color="k", lw=0.5,
                ecolor="0.6", label="PDCSAP")  # fmt: skip
    az.plot(sub["time"], full_sap + off, ".", ms=2.5, color="tab:orange", label="SAP (offset)")
    robust_ylim(az, full_pdc, full_sap + off, lo=0.2, hi=99.8, pad=0.1)
    if card.trace is not None:
        tm = (card.trace.t >= w.t0) & (card.trace.t <= w.t1)
        overlay_surprise(az, card.trace.t[tm], card.trace.v[tm])
    az.set_xlim(w.t0, w.t1)
    az.set_title(f"zoom: {w.source}", fontsize=8.5, loc="left")
    az.legend(fontsize=7, loc="lower left", markerscale=2)
    az.tick_params(labelsize=7)

    # 2b. candidate + CCD neighbours in the window
    key = "flux" if w.channel == "pdc" else "sap_flux"
    chname = "PDCSAP" if w.channel == "pdc" else "SAP"
    sg = gs[1, 1].subgridspec(n_nb + 1, 1, hspace=0.0)
    rows = [(f"this star TIC {card.tic}", lc, "k")]
    rows += [(f"TIC {nb.tic}  T={nb.tmag:.1f}  {nb.sep_arcsec / TESS_PIXEL_ARCSEC:.1f} px"
              + ("  (overlapping pixels: blend?)" if nb.sep_arcsec < 2 * TESS_PIXEL_ARCSEC
                 else ""), nb.lc, "tab:blue") for nb in card.neighbours]  # fmt: skip
    first = None
    for j, (lab, l2, col) in enumerate(rows):
        a = fig.add_subplot(sg[j, 0], sharex=first)
        first = first or a
        for tt in dumps_in:
            a.axvline(tt, color="tab:red", lw=0.6, alpha=0.6, zorder=0)
        if l2 is not None:
            mm = _in(l2, w)
            y = norm(l2[key])[mm]
            a.plot(l2["time"][mm], y, ".", ms=2.5, color=col)
            robust_ylim(a, y, lo=1, hi=99, pad=0.25)
        else:
            a.text(0.5, 0.5, "no light curve", transform=a.transAxes, ha="center", fontsize=7)
        a.text(0.005, 0.93, lab, transform=a.transAxes, fontsize=7, va="top",
               bbox={"facecolor": "white", "alpha": 0.7, "lw": 0, "pad": 1})  # fmt: skip
        a.tick_params(labelsize=6)
        a.yaxis.set_major_locator(plt.MaxNLocator(2))
        if j < len(rows) - 1:
            a.tick_params(labelbottom=False)
        if j == 0:
            a.set_title(f"same window, nearest stars on cam/CCD ({chname}, each its own scale): "
                        "shared dip = systematic", fontsize=8.5, loc="left")  # fmt: skip
    first.set_xlim(w.t0, w.t1)

    # 3. other sectors
    sg3 = gs[2, :].subgridspec(1, n_other, wspace=0.15)
    if not card.others:
        a = fig.add_subplot(sg3[0, 0])
        a.text(0.5, 0.5, "not in any other ingested sector", transform=a.transAxes,
               ha="center", va="center", fontsize=9, color="0.4")  # fmt: skip
        a.set_axis_off()
    for j, o in enumerate(card.others):
        a = fig.add_subplot(sg3[0, j])
        mark_systematics(a, o.lc["time"], o.lc["qflags"])
        _plot_lc(a, o.lc, ms=1.0)
        a.set_title(f"other sector S{o.sector} (cam{o.camera}/ccd{o.ccd})", fontsize=8,
                    loc="left")  # fmt: skip
        a.tick_params(labelsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
