"""Scorer-agnostic explain-away combiner (docs/ML_DESIGN.md §4.5).

Inputs per light curve: the base scorer's percentile ``s`` (within Tmag x CCD stratum, against
the sector's real stars) and the known-class probabilities ``p_c`` from the classifier.

**Rule: class-conditional percentile.** For each known class ``c``, ``F_c`` is the distribution
of ``s`` among the reference (train-split, real) stars *weighted by their probability of being
class c*. The explained-away score asks "how weird is this star *for the kind of star it looks
like*":

    ka = sum_c p_c * F_c(s)
    score = (1 - alpha) * s + alpha * ka          (then re-percentiled within stratum)

* A 40%-deep eclipsing binary has a high ``s``, but ordinary among stars that look like EBs, so
  ``F_EB(s)`` is ~0.5 and it drops.
* A quiet star with an injected 0.3% dip keeps ``F_quiet(s)`` near 1: quiet stars almost never
  score that high, so the quiet class cannot explain the dip away. This is why "quiet" is a
  class and not just "no class".
* An EB host with an injected dip is judged against other EBs: it stays findable only if the
  dip raises its base score above typical EBs. The classifier sees the injected light curve
  (never the host's catalogue label), so real and injected stars are treated identically and a
  dip that makes the light curve look less EB-like also loosens the class match.
* A star that fits no class (spread probabilities) gets an average of several ``F_c``, so
  it is only partly explained away.

``alpha`` is the only knob. It is chosen on **val** injections (``choose_alpha``), never on the
test split or the known positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oddball.score.classical import StratumECDF


class WeightedECDF:
    """Mid-rank ECDF of values with per-sample weights (one per class column)."""

    def fit(self, values: np.ndarray, weights: np.ndarray) -> WeightedECDF:
        v = np.where(np.isfinite(values), values, -np.inf)
        o = np.argsort(v, kind="stable")
        self.v = v[o]
        w = np.asarray(weights, np.float64)[o]  # (n, K)
        self.cw = np.vstack([np.zeros((1, w.shape[1])), np.cumsum(w, axis=0)])
        self.tot = self.cw[-1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """(n, K): for each query value, its weighted mid-rank percentile in every class."""
        x = np.where(np.isfinite(x), x, -np.inf)
        lo = np.searchsorted(self.v, x, side="left")
        hi = np.searchsorted(self.v, x, side="right")
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.cw[lo] + self.cw[hi]) / 2.0 / self.tot[None, :]


class ExplainAway:
    def __init__(self, classes: list[str]):
        self.classes = list(classes)

    def fit(self, s_ref: np.ndarray, p_ref: np.ndarray) -> ExplainAway:
        self.ecdf = WeightedECDF().fit(np.asarray(s_ref, float), np.asarray(p_ref, float))
        self.class_mass = dict(zip(self.classes, self.ecdf.tot.tolist(), strict=True))
        return self

    def ka(self, s: np.ndarray, p: np.ndarray) -> np.ndarray:
        f = self.ecdf.transform(np.asarray(s, float))
        f = np.nan_to_num(f, nan=0.5)  # a class with zero reference mass explains nothing
        return (np.asarray(p, float) * f).sum(axis=1)

    def score(self, s: np.ndarray, p: np.ndarray, alpha: float) -> np.ndarray:
        s = np.asarray(s, float)
        return (1.0 - alpha) * np.nan_to_num(s, nan=0.0) + alpha * self.ka(s, p)


def stratum_pct(ref_scores, ref_strata, q_scores, q_strata) -> np.ndarray:
    """Percentile of each query score among reference scores of its stratum (mid-rank)."""
    e = StratumECDF().fit(np.asarray(ref_scores, float)[:, None], np.asarray(ref_strata))
    return e.transform(np.asarray(q_scores, float)[:, None], np.asarray(q_strata))[:, 0]


def prob_matrix(kc: pd.DataFrame, classes: list[str]) -> np.ndarray:
    return kc[[f"kc_p_{c}" for c in classes]].to_numpy(np.float64)


def choose_alpha(metric_by_alpha: dict[float, float]) -> float:
    """Best alpha by the val metric; ties go to the smaller alpha (less intervention)."""
    best = max(metric_by_alpha.values())
    return min(a for a, m in metric_by_alpha.items() if m >= best - 1e-12)
