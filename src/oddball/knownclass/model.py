"""Known-class classifier (docs/ML_DESIGN.md §4.5): labels -> per-light-curve training targets,
a gradient-boosted tree model over any feature table, temperature calibration on val, and the
per-star outputs (predicted class, max probability, normalised entropy, energy).

Training targets are per light curve (``tic, sector``):

* a catalogue class (``labels.parquet``) is used only when that sector's light curve actually
  varies (``lc_variable``): a long-period EB with no eclipse in the sector, or a variable lost
  in the noise of a faint star, would otherwise teach "flat light curve = EB";
* ``quiet`` = no catalogue flags at all (not excluded, no VSX entry within one TESS pixel, no
  Prsa EB, no Gaia variability) **and** a light curve consistent with noise (``lc_quiet``);
* everything else (unlabelled variables, excluded stars) is not a target.

Splits come only from ``oddball.common.splits``: fit on ``train``, early stopping and
temperature on ``val``. ``known`` (known positives) and ``test`` are never used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oddball.knownclass.labels import CLASSES

# Light-curve variability gates (PDCSAP features; see features.py / classical.py)
QUIET_MAX_VAR_FRAC = 0.15  # fraction of variance above the point-to-point white noise
QUIET_MAX_SNR = 6.0  # largest dip / bump box SNR (pure noise gives ~4)
VAR_MIN_VAR_FRAC = 0.3
VAR_MIN_DIP_SNR = 10.0
QUIET_PER_CLASS_CAP = 3.0  # quiet rows are subsampled to <= this x the largest variable class


def lc_quiet(f: pd.DataFrame) -> np.ndarray:
    return ((f.pdc_var_frac.to_numpy() < QUIET_MAX_VAR_FRAC)
            & (np.nan_to_num(f.pdc_dip_snr.to_numpy(), nan=99) < QUIET_MAX_SNR)
            & (np.nan_to_num(f.pdc_bump_snr.to_numpy(), nan=99) < QUIET_MAX_SNR))  # fmt: skip


def lc_variable(f: pd.DataFrame) -> np.ndarray:
    return ((np.nan_to_num(f.pdc_var_frac.to_numpy()) > VAR_MIN_VAR_FRAC)
            | (np.nan_to_num(f.pdc_dip_snr.to_numpy()) > VAR_MIN_DIP_SNR))  # fmt: skip


def training_targets(feat: pd.DataFrame, labels: pd.DataFrame, seed: int = 0) -> pd.Series:
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
    if feat.inj_id.notna().any():
        y[feat.inj_id.notna().to_numpy()] = ""  # injected light curves are never targets
    return pd.Series(y, index=feat.index, name="y")


def cap_quiet(y: pd.Series, cap: float = QUIET_PER_CLASS_CAP, seed: int = 0) -> pd.Series:
    """Subsample 'quiet' rows to ``cap`` x the largest other class (deterministic)."""
    counts = y[(y != "") & (y != "quiet")].value_counts()
    if counts.empty:
        return y
    n_max = int(cap * counts.max())
    q = np.flatnonzero(y.to_numpy() == "quiet")
    if len(q) <= n_max:
        return y
    rng = np.random.default_rng(seed)
    drop = rng.choice(q, size=len(q) - n_max, replace=False)
    y = y.copy()
    y.iloc[drop] = ""
    return y


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

    def predict_frame(self, X: pd.DataFrame, batch: int = 50000) -> pd.DataFrame:
        """Per row: kc_pred, kc_pmax, kc_entropy (normalised 0-1), kc_energy, kc_p_<class>."""
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
            parts.append(pd.DataFrame(d))
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
