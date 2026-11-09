"""Multiplicative flux models for injection.

Every injection is a set of *components* (unit-peak shapes ``s_i(t)`` with relative amplitudes
``a_i``) combined as overlapping occulters::

    S(t) = 1 - prod_i (1 - a_i s_i(t)),   S /= norm  (so max S = 1)
    m(t) = 1 + sign * depth * S(t)        sign = -1 dips, +1 brightenings

so ``depth`` is exactly the peak fractional flux change of the native-resolution model. The
injected light curve is ``flux * m`` (see apply.py). The whole model is rebuilt from
``params`` alone (``model_from_params``), so ``params_json`` in ``injections.parquet`` is a
complete, auditable record.

Shapes (all peak at ``tc`` with value 1):

* ``gauss``  symmetric Gaussian, ``fwhm``.
* ``agauss`` two-sided Gaussian: ``fwhm_in`` before the peak, ``fwhm_eg`` after.
* ``fred``   fast rise, exponential decay. ``rise`` = time from half-max to peak on the
  rising side; ``rise_kind='linear'`` is a straight ramp of length ``2*rise`` (a sharp
  exocomet ingress), ``'gauss'`` a half-Gaussian (an outburst rise). After the peak
  ``exp(-(t-tc)/tau)``.
* ``box``    trapezoid: flat bottom of total width ``width``, linear edges of ``edge`` each.
* ``ramp``   linear decline over ``width`` to the peak, then an instant recovery.
"""

from __future__ import annotations

import numpy as np

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
LN2 = float(np.log(2.0))


def shape_gauss(t, tc, fwhm):
    s = fwhm * FWHM_TO_SIGMA
    return np.exp(-0.5 * ((t - tc) / s) ** 2)


def shape_agauss(t, tc, fwhm_in, fwhm_eg):
    s = np.where(t < tc, fwhm_in, fwhm_eg) * FWHM_TO_SIGMA
    return np.exp(-0.5 * ((t - tc) / s) ** 2)


def shape_fred(t, tc, rise, tau, rise_kind="linear"):
    dt = t - tc
    after = np.exp(-np.clip(dt, 0.0, None) / tau)
    if rise_kind == "linear":
        before = np.clip(1.0 + dt / (2.0 * rise), 0.0, 1.0)
    elif rise_kind == "gauss":
        sig = rise / np.sqrt(2.0 * LN2)  # half-max reached `rise` before the peak
        before = np.exp(-0.5 * (np.clip(dt, None, 0.0) / sig) ** 2)
    else:
        raise ValueError(rise_kind)
    return np.where(dt < 0, before, after)


def shape_box(t, tc, width, edge):
    half = width / 2.0
    d = np.abs(t - tc)
    return np.clip((half + edge - d) / edge, 0.0, 1.0) if edge > 0 else (d <= half).astype(float)


def shape_ramp(t, tc, width):
    dt = t - tc
    return np.where((dt <= 0) & (dt > -width), 1.0 + dt / width, 0.0)


SHAPES = {
    "gauss": (shape_gauss, ("tc", "fwhm")),
    "agauss": (shape_agauss, ("tc", "fwhm_in", "fwhm_eg")),
    "fred": (shape_fred, ("tc", "rise", "tau", "rise_kind")),
    "box": (shape_box, ("tc", "width", "edge")),
    "ramp": (shape_ramp, ("tc", "width")),
}


def component(c: dict, t: np.ndarray) -> np.ndarray:
    fn, keys = SHAPES[c["shape"]]
    return fn(t, *[c[k] for k in keys if k in c])


def raw_profile(components: list[dict], t: np.ndarray) -> np.ndarray:
    """Unnormalised S(t) = 1 - prod(1 - a_i s_i)."""
    keep = np.ones_like(t, dtype=np.float64)
    for c in components:
        keep *= 1.0 - float(c.get("amp", 1.0)) * component(c, t)
    return 1.0 - keep


def fine_grid(t_lo: float, t_hi: float, step_d: float = 2.0 / 1440.0) -> np.ndarray:
    return np.arange(t_lo, t_hi + step_d, step_d)


def normalise(components: list[dict], t_lo: float, t_hi: float) -> tuple[float, float]:
    """Return ``(norm, t_peak)``: max of the raw profile over a 2-min grid, and where it is."""
    tf = fine_grid(t_lo, t_hi)
    s = raw_profile(components, tf)
    i = int(np.argmax(s))
    return float(s[i]), float(tf[i])


def model_from_params(params: dict, t: np.ndarray) -> np.ndarray:
    """Multiplicative model m(t) at instants ``t`` (BTJD)."""
    t = np.asarray(t, dtype=np.float64)
    s = raw_profile(params["components"], t) / params["norm"]
    return 1.0 + params["sign"] * params["depth"] * s


def binned_model(params: dict, t_centres: np.ndarray, bin_days: float, nsub: int = 5):
    """Model averaged over each bin (``nsub`` sub-samples; 30-min bins -> 6-min sampling).

    The stored flux is the mean of native 10-min cadences in a bin, so this is the model the
    binned light curve actually sees.
    """
    t_centres = np.asarray(t_centres, dtype=np.float64)
    off = (np.arange(nsub) + 0.5) / nsub - 0.5
    tt = t_centres[:, None] + off[None, :] * bin_days
    return model_from_params(params, tt.ravel()).reshape(tt.shape).mean(axis=1)
