"""Outburst (sustained brightening) detector v0 for Rubin light curves.

Pure functions on arrays; no I/O. The unit of evidence is a *night* (Rubin usually takes
two visits per field per night, so a same-night pair is not independent confirmation).

Per band:
1. Bin visits into nights (median mag).
2. Baseline = median of the faintest ``baseline_quantile`` of nights (>= ``min_baseline_nights``),
   scatter = 1.4826 * MAD of those nights. Using the faint end keeps the quiescent level even
   when the star is in outburst for a large part of the season.
3. A night is *bright* if Delta = baseline - mag >= ``dmag_min`` and Delta is significant
   (Delta / sqrt(err^2 + scatter^2) >= ``snr_min``).
4. The band triggers if it has >= ``min_nights`` *consecutive* bright nights (data gaps do not
   break a run; an intervening non-bright night does). A one-night spike never triggers.
5. Dipper / flicker guard: if quiet nights (Delta < dmag_min/2) are interleaved inside the span
   of bright nights for more than ``max_interleave`` of that span, reject. An outburst is one
   contiguous bright block; a dipper's "bright" state is its normal state with dips mixed in.

Per object: the band with the strongest run is primary; every other band with nights inside
the run window must also brighten (>= ``xband_min``) or the object is rejected as
band-inconsistent (a real accretion outburst brightens in all optical bands, more in g).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

ZP_NJY = 31.4


@dataclass
class DetectConfig:
    dmag_min: float = 1.0
    snr_min: float = 5.0
    min_nights: int = 2
    min_baseline_nights: int = 3
    baseline_quantile: float = 0.25
    night_offset: float = 0.667  # MJD fraction of local noon at Cerro Pachon (~16 UT)
    max_interleave: float = 0.1
    min_run_fraction: float = 0.5
    rise_level: float = 0.8
    xband_min: float = 0.5
    xband_contradict: float = 0.25
    xband_window_days: float = 2.0
    saturation_mag: float = 16.5  # Rubin 30 s visits saturate around 16-17 mag
    scatter_floor: float = 0.02
    template_scatter_floor: float = 0.05
    use_template: bool = True


@dataclass
class BandResult:
    band: str
    status: str  # "triggered", "quiet", "too_few", "interleaved", "fragmented"
    n_nights: int = 0
    baseline_mag: float = np.nan
    baseline_scatter: float = np.nan
    peak_mag: float = np.nan
    peak_dmag: float = np.nan
    t_peak: float = np.nan
    run_len: int = 0
    t_run_start: float = np.nan
    t_run_end: float = np.nan
    rise_time: float = np.nan  # upper bound: last quiet night -> first night at rise_level
    rise_time_lo: float = np.nan  # lower bound: first non-quiet night -> same
    rise_resolved: bool = False
    interleave: float = np.nan
    n_bright: int = 0
    baseline_src: str = ""  # "lc" (faint end of the light curve) or "template"


@dataclass
class ObjectResult:
    flagged: bool
    reasons: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    primary: BandResult | None = None
    bands: dict = field(default_factory=dict)
    xband: dict = field(default_factory=dict)
    diff_ratio: float = np.nan

    def summary(self) -> dict:
        p = self.primary
        out = {"flagged": self.flagged, "reasons": ";".join(self.reasons),
               "flags": ";".join(self.flags), "diff_ratio": self.diff_ratio,
               "xband": ";".join(f"{b}:{v:+.2f}" for b, v in self.xband.items())}
        if p is not None:
            d = asdict(p)
            out.update({f"p_{k}": v for k, v in d.items()})
        return out


def bin_nights(t, mag, err, offset: float = 0.667):
    """Median-combine visits per night. Returns (t_night, mag, err) sorted in time."""
    t = np.asarray(t, float)
    mag = np.asarray(mag, float)
    err = np.asarray(err, float)
    ok = np.isfinite(t) & np.isfinite(mag) & np.isfinite(err)
    t, mag, err = t[ok], mag[ok], err[ok]
    if t.size == 0:
        return t, mag, err
    night = np.floor(t - offset).astype(np.int64)
    order = np.argsort(t)
    t, mag, err, night = t[order], mag[order], err[order], night[order]
    _, start = np.unique(night, return_index=True)
    bounds = list(start) + [t.size]
    tn, mn, en = [], [], []
    for a, b in itertools.pairwise(bounds):
        tn.append(t[a:b].mean())
        mn.append(np.median(mag[a:b]))
        en.append(np.median(err[a:b]) / np.sqrt(b - a))
    return np.array(tn), np.array(mn), np.array(en)


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """(start, length) of the longest run of True; (0, 0) if none."""
    best_s, best_n, s, n = 0, 0, 0, 0
    for i, v in enumerate(mask):
        if v:
            if n == 0:
                s = i
            n += 1
            if n > best_n:
                best_s, best_n = s, n
        else:
            n = 0
    return best_s, best_n


def baseline_of(mag: np.ndarray, cfg: DetectConfig) -> tuple[float, float]:
    k = max(cfg.min_baseline_nights, int(np.ceil(cfg.baseline_quantile * mag.size)))
    faint = np.sort(mag)[::-1][:k]
    base = float(np.median(faint))
    scatter = max(1.4826 * float(np.median(np.abs(faint - base))), cfg.scatter_floor)
    return base, scatter


def detect_band(t, mag, err, band: str = "?", cfg: DetectConfig | None = None,
                ref_mag: float = np.nan, ref_err: float = np.nan) -> BandResult:
    """``ref_mag``: template (reference-image) magnitude at the object position, if known.

    Alert detections only exist when the star differs from the template, so an outburst
    from quiescence may have *no* quiescent epochs in the light curve. The template then
    supplies the baseline. When both exist the fainter one is used (the quiescent level).
    """
    cfg = cfg or DetectConfig()
    tn, mn, en = bin_nights(t, mag, err, cfg.night_offset)
    r = BandResult(band=band, status="too_few", n_nights=int(tn.size))
    has_ref = cfg.use_template and np.isfinite(ref_mag)
    has_lc = tn.size >= cfg.min_baseline_nights + cfg.min_nights
    if not has_lc and not (has_ref and tn.size >= cfg.min_nights):
        return r
    base, scatter, src = np.nan, np.nan, ""
    if has_lc:
        base, scatter = baseline_of(mn, cfg)
        src = "lc"
    if has_ref and not (ref_mag <= base):  # template fainter (or no LC baseline)
        rerr = ref_err if np.isfinite(ref_err) else 0.0
        base, scatter, src = float(ref_mag), max(float(rerr), cfg.template_scatter_floor), \
            "template"
    r.baseline_mag, r.baseline_scatter, r.baseline_src = base, scatter, src
    delta = base - mn
    sig = delta / np.sqrt(en**2 + scatter**2)
    bright = (delta >= cfg.dmag_min) & (sig >= cfg.snr_min)
    r.n_bright = int(bright.sum())
    s, n = _longest_run(bright)
    r.run_len = n
    if n < cfg.min_nights:
        r.status = "quiet"
        i = int(np.argmax(delta))
        r.peak_mag, r.peak_dmag, r.t_peak = float(mn[i]), float(delta[i]), float(tn[i])
        return r
    run = slice(s, s + n)
    i_peak = s + int(np.argmax(delta[run]))
    r.peak_mag, r.peak_dmag, r.t_peak = float(mn[i_peak]), float(delta[i_peak]), float(tn[i_peak])
    r.t_run_start, r.t_run_end = float(tn[s]), float(tn[s + n - 1])
    # rise: from the last quiet night before the run to the first run night reaching
    # rise_level * peak Delta (robust on a noisy plateau, unlike argmax timing)
    quiet = delta < cfg.dmag_min / 2
    i_hi = s + int(np.argmax(delta[run] >= cfg.rise_level * r.peak_dmag))
    pre = np.nonzero(quiet[:s])[0]
    if pre.size:
        r.rise_time = float(tn[i_hi] - tn[pre[-1]])
        r.rise_time_lo = float(tn[i_hi] - tn[pre[-1] + 1])
        r.rise_resolved = True
    else:  # already bright at the first night: only a lower limit
        r.rise_time = r.rise_time_lo = float(tn[i_hi] - tn[s])
    # interleave: quiet nights inside the span of all bright nights
    ib = np.nonzero(bright)[0]
    span = slice(ib[0], ib[-1] + 1)
    n_span = ib[-1] - ib[0] + 1
    r.interleave = float(quiet[span].sum() / n_span)
    if r.interleave > cfg.max_interleave:
        r.status = "interleaved"
    elif n < cfg.min_run_fraction * r.n_bright:
        r.status = "fragmented"  # bright nights scattered in short runs, not one block
    else:
        r.status = "triggered"
    return r


def detect_object(lc: pd.DataFrame, cfg: DetectConfig | None = None) -> ObjectResult:
    """``lc`` needs columns mjd, band, mag, magerr; optional bad (bool), diff_flux, sci_flux."""
    cfg = cfg or DetectConfig()
    df = lc
    if "bad" in df.columns:
        df = df[~df["bad"].astype(bool)]
    df = df[np.isfinite(df["mag"]) & np.isfinite(df["magerr"])]
    res = ObjectResult(flagged=False)
    if df.empty:
        res.reasons.append("no_good_photometry")
        return res
    for b, g in df.groupby("band"):
        ref_mag = ref_err = np.nan
        if "template_flux" in g.columns:
            tf = g["template_flux"].to_numpy(float)
            te = g["template_flux_err"].to_numpy(float) if "template_flux_err" in g.columns \
                else np.full(tf.size, np.nan)
            ok = np.isfinite(tf) & (tf > 0)
            if ok.any():
                f = float(np.median(tf[ok]))
                ref_mag = ZP_NJY - 2.5 * np.log10(f)
                ref_err = 1.0857362 * float(np.median(te[ok])) / f if np.isfinite(te[ok]).any() \
                    else np.nan
        res.bands[b] = detect_band(g["mjd"].to_numpy(), g["mag"].to_numpy(),
                                   g["magerr"].to_numpy(), band=b, cfg=cfg,
                                   ref_mag=ref_mag, ref_err=ref_err)
    trig = [r for r in res.bands.values() if r.status == "triggered"]
    inter = [r for r in res.bands.values() if r.status in ("interleaved", "fragmented")]
    if not trig:
        res.reasons.append("interleaved_quiet_nights" if inter else "no_sustained_brightening")
        cands = inter or [r for r in res.bands.values() if np.isfinite(r.peak_dmag)]
        if cands:
            res.primary = max(cands, key=lambda r: r.peak_dmag)
        return res
    p = max(trig, key=lambda r: (r.run_len, r.peak_dmag))
    res.primary = p
    if inter:
        res.reasons.append("interleaved_quiet_nights:" + ",".join(r.band for r in inter))
    # cross-band consistency inside the run window
    w0, w1 = p.t_run_start - cfg.xband_window_days, p.t_run_end + cfg.xband_window_days
    contradict = []
    for b, r in res.bands.items():
        if b == p.band or not np.isfinite(r.baseline_mag):
            continue
        g = df[(df["band"] == b) & (df["mjd"] >= w0) & (df["mjd"] <= w1)]
        if g.empty:
            continue
        d = r.baseline_mag - float(np.median(g["mag"]))
        res.xband[b] = d
        if d < cfg.xband_contradict:
            contradict.append(b)
    if contradict:
        res.reasons.append("band_inconsistent:" + ",".join(contradict))
    if not res.xband:
        res.flags.append("single_band")
    elif all(v >= cfg.xband_min for v in res.xband.values()):
        res.flags.append("multiband_confirmed")
    # saturation
    if p.baseline_mag < cfg.saturation_mag:
        res.reasons.append("baseline_saturated")
    elif p.peak_mag < cfg.saturation_mag:
        res.flags.append("peak_near_saturation")
    if "saturated" in lc.columns:
        win = lc[(lc["band"] == p.band) & (lc["mjd"] >= p.t_run_start - 0.5)
                 & (lc["mjd"] <= p.t_run_end + 0.5)]
        if win["saturated"].astype(bool).any():
            res.flags.append("saturated_pixels_in_run")
    # difference-flux consistency: change in diff flux should track change in science flux
    if {"diff_flux", "sci_flux"} <= set(df.columns):
        g = df[df["band"] == p.band]
        in_run = (g["mjd"] >= p.t_run_start - 0.5) & (g["mjd"] <= p.t_run_end + 0.5)
        quiet = g["mag"] > p.baseline_mag - cfg.dmag_min / 2
        if in_run.any() and quiet.any():
            d_sci = g.loc[in_run, "sci_flux"].median() - g.loc[quiet, "sci_flux"].median()
            d_dif = g.loc[in_run, "diff_flux"].median() - g.loc[quiet, "diff_flux"].median()
            if np.isfinite(d_sci) and d_sci > 0 and np.isfinite(d_dif):
                res.diff_ratio = float(d_dif / d_sci)
                if not 0.3 <= res.diff_ratio <= 3.0:
                    res.flags.append("diff_flux_inconsistent")
    if not p.rise_resolved:
        res.flags.append("rise_unresolved")
    if p.baseline_src == "template":
        res.flags.append("template_baseline")
    res.flagged = not res.reasons
    return res
