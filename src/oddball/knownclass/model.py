"""Known-class classifier (docs/ML_DESIGN.md §4.5): labels -> per-light-curve training targets,
a gradient-boosted tree model over any feature table, temperature calibration on val, and the
per-star outputs (predicted class, max probability, normalised entropy, energy).

Training targets are per light curve (``tic, sector``):

* a catalogue class (``labels.parquet``) is used only when that sector's light curve actually
  varies (``lc_variable``): a long-period EB with no eclipse in the sector, or a variable lost
  in the noise of a faint star, would otherwise teach "flat light curve = EB";
* ``quiet`` = no catalogue flags at all (not excluded, no VSX entry within one TESS pixel, no
  Prsa EB, no Gaia variability) **and** a light curve consistent with noise (``lc_quiet``);
* optional ``other`` (``training_targets(other=True)``, an ablation): not excluded, no usable
  catalogue class, not quiet. It never explains anything away.
* everything else (catalogued stars flat in this sector, excluded stars, injections) is not a
  target.

**Closed set + typicality.** A closed-set classifier pushes every light curve into some known
class with high confidence (dry run: 0.1%-deep injections were "ROT/DSCT/EB" at median
p = 0.94; trees extrapolate flat). The guard is *typicality*: a kNN check, in shape-feature space
only, of how far a light curve sits from the training members of each class, calibrated on the
members' own distances (``fit_typicality``). ``kc_w_c = p_c x t_c`` is the explaining weight.
On S32 val injections (dry run, partial labels) this beat the ``other`` catch-all: objective
0.174 vs 0.155 (mean val recall in the top 1% at ~0.3/1/3% depth; base 0.127).

Splits come only from ``oddball.common.splits``: fit on ``train``, early stopping and
temperature on ``val``. ``known`` (known positives) and ``test`` are never used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oddball.knownclass.labels import CLASSES, OTHER

# Light-curve variability gates (PDCSAP features; see features.py / classical.py)
QUIET_MAX_VAR_FRAC = 0.15  # fraction of variance above the point-to-point white noise
QUIET_MAX_SNR = 6.0  # largest dip / bump box SNR (pure noise gives ~4)
VAR_MIN_VAR_FRAC = 0.3
VAR_MIN_DIP_SNR = 10.0
QUIET_PER_CLASS_CAP = 3.0  # quiet / other rows: <= this x the largest catalogue class


def lc_quiet(f: pd.DataFrame) -> np.ndarray:
    return ((f.pdc_var_frac.to_numpy() < QUIET_MAX_VAR_FRAC)
            & (np.nan_to_num(f.pdc_dip_snr.to_numpy(), nan=99) < QUIET_MAX_SNR)
            & (np.nan_to_num(f.pdc_bump_snr.to_numpy(), nan=99) < QUIET_MAX_SNR))  # fmt: skip


def lc_variable(f: pd.DataFrame) -> np.ndarray:
    return ((np.nan_to_num(f.pdc_var_frac.to_numpy()) > VAR_MIN_VAR_FRAC)
            | (np.nan_to_num(f.pdc_dip_snr.to_numpy()) > VAR_MIN_DIP_SNR))  # fmt: skip


def training_targets(feat: pd.DataFrame, labels: pd.DataFrame, other: bool = False) -> pd.Series:
    """Per-row target class ('' = not a target) for real light curves of a feature table."""
    lab = labels.set_index("tic")
    cat = lab.label.reindex(feat.tic).fillna("").to_numpy(object)
    excl = lab.excluded.reindex(feat.tic).fillna(True).to_numpy(bool)
    flagged = lab.flagged_any.reindex(feat.tic).fillna(True).to_numpy(bool)
    y = np.full(len(feat), "", dtype=object)
    var_ok = lc_variable(feat)
    is_cat = (cat != "") & ~excl
    y[is_cat & var_ok] = cat[is_cat & var_ok]
    quiet = ~excl & ~flagged & (cat == "") & lc_quiet(feat)
    y[quiet] = "quiet"
    if other:
        y[~excl & (cat == "") & ~quiet] = OTHER
    if feat.inj_id.notna().any():
        y[feat.inj_id.notna().to_numpy()] = ""  # injected light curves are never targets
    return pd.Series(y, index=feat.index, name="y")


def cap_quiet(y: pd.Series, cap: float = QUIET_PER_CLASS_CAP, seed: int = 0,
              bulk: tuple[str, ...] = ("quiet", OTHER)) -> pd.Series:  # fmt: skip
    """Subsample each ``bulk`` class to ``cap`` x the largest catalogue class (deterministic)."""
    yv = y.to_numpy()
    counts = y[(y != "") & ~np.isin(yv, bulk)].value_counts()
    if counts.empty:
        return y
    n_max = int(cap * counts.max())
    y = y.copy()
    rng = np.random.default_rng(seed)
    for cls in bulk:
        q = np.flatnonzero(yv == cls)
        if len(q) > n_max:
            y.iloc[rng.choice(q, size=len(q) - n_max, replace=False)] = ""
    return y


NON_EXPLAINING = ("quiet", OTHER)  # classes that never explain an event away
TYP_K = 10
TYP_MAX_PER_CLASS = 5000
TYP_MIN_MEMBERS = 15
# Typicality measures *shape* (periodicity, profile, event count, asymmetry), not amplitude:
# a 40%-deep EB is still a typical EB, while one dip on a flat star is not. Amplitude, SNR and
# stellar-context columns are left out of the typicality space (the classifier still sees them).
TYP_AMPLITUDE_TOKENS = ("dip_snr", "dip_depth", "bump_snr", "xs_var", "sigma_w", "log_amp",
                        "amp9005", "frac_neg3", "ctx_")


def typicality_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if not any(t in c for t in TYP_AMPLITUDE_TOKENS)]


def _knn_mean_dist(q: np.ndarray, ref: np.ndarray, k: int, q_tic=None, r_tic=None,
                   chunk: int = 2048) -> np.ndarray:  # fmt: skip
    """Mean Euclidean distance from each query row to its ``k`` nearest reference rows,
    skipping reference rows with the query's own TIC (brute force, chunked)."""
    out = np.full(len(q), np.nan)
    rn = (ref.astype(np.float64) ** 2).sum(1)
    kk = min(k, len(ref))
    for s in range(0, len(q), chunk):
        qq = q[s : s + chunk].astype(np.float64)
        d2 = (qq**2).sum(1)[:, None] + rn[None, :] - 2.0 * qq @ ref.T.astype(np.float64)
        if q_tic is not None and r_tic is not None:
            d2[np.asarray(q_tic[s : s + chunk])[:, None] == np.asarray(r_tic)[None, :]] = np.inf
        part = np.partition(d2, kk - 1, axis=1)[:, :kk]
        out[s : s + chunk] = np.sqrt(np.maximum(part, 0)).mean(1)
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _logsumexp(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1)
    return m + np.log(np.exp(z - m[:, None]).sum(axis=1))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising the NLL of ``softmax(logits / T)`` (golden-section on log T)."""

    def nll(logt):
        z = logits / np.exp(logt)
        return float(np.mean(_logsumexp(z) - z[np.arange(len(y)), y]))

    a, b = np.log(0.05), np.log(20.0)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(60):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(np.exp((a + b) / 2))


@dataclass
class KnownClassModel:
    feature_cols: list[str]
    classes: list[str] = field(default_factory=list)
    temperature: float = 1.0
    params: dict = field(default_factory=dict)
    clf: object = None

    def fit(self, Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray,
            class_weight_power: float = 0.5, seed: int = 0) -> KnownClassModel:  # fmt: skip
        from sklearn.ensemble import HistGradientBoostingClassifier

        present = set(ytr)
        self.classes = [c for c in CLASSES if c in present]
        idx = {c: i for i, c in enumerate(self.classes)}
        ytr_i = np.array([idx[c] for c in ytr])
        keep_va = np.isin(yva, self.classes)
        yva_i = np.array([idx[c] for c in yva[keep_va]])
        cnt = np.bincount(ytr_i, minlength=len(self.classes)).astype(float)
        w_cls = (cnt.max() / cnt) ** class_weight_power  # tempered inverse frequency
        p = {"max_iter": 500, "learning_rate": 0.08, "max_leaf_nodes": 31, "min_samples_leaf": 40,
                 "l2_regularization": 1.0, "early_stopping": True, "n_iter_no_change": 25,
                 "scoring": "loss", "random_state": seed}  # fmt: skip
        p.update(self.params)
        self.params = p
        self.clf = HistGradientBoostingClassifier(**p)
        self.clf.fit(Xtr[self.feature_cols].to_numpy(np.float32), ytr_i,
                     sample_weight=w_cls[ytr_i],
                     X_val=Xva[self.feature_cols].to_numpy(np.float32)[keep_va], y_val=yva_i,
                     sample_weight_val=w_cls[yva_i])  # fmt: skip
        logits = self.logits(Xva[keep_va])
        self.temperature = fit_temperature(logits, yva_i)
        self.class_weights = dict(zip(self.classes, w_cls.tolist(), strict=True))
        return self

    def logits(self, X: pd.DataFrame) -> np.ndarray:
        z = self.clf.decision_function(X[self.feature_cols].to_numpy(np.float32))
        return z if z.ndim == 2 else np.stack([-z, z], axis=1)

    def fit_typicality(self, X: pd.DataFrame, y: np.ndarray, tic: np.ndarray, k: int = TYP_K,
                       max_per_class: int = TYP_MAX_PER_CLASS, seed: int = 0) -> KnownClassModel:
        """Per explaining class: kNN reference of its training members, for ``typicality``.

        Shape features (``typicality_columns``) are quantile-transformed to N(0, 1) (NaN -> 0). A member's own distance scale
        is its mean distance to its ``k`` nearest *other-TIC* members (the same star in another
        sector is a near-duplicate), which gives the class's distance distribution.
        """  # fmt: skip
        from sklearn.preprocessing import QuantileTransformer

        rng = np.random.default_rng(seed)
        self.qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                      subsample=200_000, random_state=seed)  # fmt: skip
        self.typ_cols = typicality_columns(self.feature_cols)
        self.qt.fit(X[self.typ_cols].to_numpy(np.float32))
        self.typ = {}
        for c in self.classes:
            if c in NON_EXPLAINING:
                continue
            idx = np.flatnonzero(y == c)
            if len(idx) < TYP_MIN_MEMBERS:
                continue
            if len(idx) > max_per_class:
                idx = rng.choice(idx, max_per_class, replace=False)
            ref = self._z(X.iloc[idx])
            d_self = _knn_mean_dist(ref, ref, k, q_tic=tic[idx], r_tic=tic[idx])
            self.typ[c] = {"ref": ref.astype(np.float32), "tic": tic[idx],
                           "dist": np.sort(d_self[np.isfinite(d_self)])}  # fmt: skip
        self.typ_k = k
        return self

    def _z(self, X: pd.DataFrame) -> np.ndarray:
        z = self.qt.transform(X[self.typ_cols].to_numpy(np.float32))
        return np.nan_to_num(z, nan=0.0)

    def typicality(self, X: pd.DataFrame, tic: np.ndarray | None = None) -> pd.DataFrame:
        """kc_t_<class>: share of the class's own members that sit farther from their k nearest
        fellow members than this light curve does from its k nearest members (1 = core of the
        class, ~0 = outside it). ``tic``: exclude these TICs from each query's neighbours."""
        z = self._z(X)
        out = {}
        for c, t in self.typ.items():
            d = _knn_mean_dist(z, t["ref"], self.typ_k, q_tic=tic, r_tic=t["tic"])
            r = t["dist"]
            out[f"kc_t_{c}"] = 1.0 - np.searchsorted(r, d, side="left") / max(len(r), 1)
        return pd.DataFrame(out, index=X.index)

    def predict_frame(self, X: pd.DataFrame, batch: int = 50000,
                      tic: np.ndarray | None = None) -> pd.DataFrame:  # fmt: skip
        """Per row: kc_pred, kc_pmax, kc_entropy (normalised 0-1), kc_energy, kc_p_<class>;
        with a typicality reference also kc_t_<class>, kc_w_<class> = p * t and
        ``kc_explained`` = sum of kc_w over the explaining classes (all but quiet / other)."""
        parts = []
        for s in range(0, len(X), batch):
            z = self.logits(X.iloc[s : s + batch]) / self.temperature
            p = _softmax(z)
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -(np.where(p > 0, p * np.log(p), 0.0)).sum(1) / np.log(len(self.classes))
            d = {"kc_pred": np.asarray(self.classes, dtype=object)[p.argmax(1)],
                 "kc_pmax": p.max(1), "kc_entropy": ent,
                 "kc_energy": -self.temperature * _logsumexp(z)}  # fmt: skip
            for j, c in enumerate(self.classes):
                d[f"kc_p_{c}"] = p[:, j]
            part = pd.DataFrame(d)
            if getattr(self, "typ", None):
                tt = self.typicality(X.iloc[s : s + batch],
                                     None if tic is None else tic[s : s + batch])  # fmt: skip
                expl = np.zeros(len(part))
                for c in self.typ:
                    tc = tt[f"kc_t_{c}"].to_numpy()
                    part[f"kc_t_{c}"] = tc
                    part[f"kc_w_{c}"] = part[f"kc_p_{c}"].to_numpy() * tc
                    expl += part[f"kc_w_{c}"].to_numpy()
                part["kc_explained"] = expl
            parts.append(part)
        out = pd.concat(parts, ignore_index=True)
        out.index = X.index
        return out


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    rows = []
    for c in classes:
        t = y_true == c
        pr = y_pred == c
        rows.append({"class": c, "n": int(t.sum()),
                     "recall": float((pr & t).sum() / max(t.sum(), 1)),
                     "precision": float((pr & t).sum() / max(pr.sum(), 1))})  # fmt: skip
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"class": "ALL (accuracy)", "n": len(y_true),
                       "recall": float((y_true == y_pred).mean()), "precision": np.nan}
    return df


def confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    m = pd.crosstab(pd.Categorical(y_true, classes), pd.Categorical(y_pred, classes),
                    rownames=["true"], colnames=["pred"], dropna=False)  # fmt: skip
    return m
