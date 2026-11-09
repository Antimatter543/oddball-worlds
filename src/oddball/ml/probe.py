"""Nuisance probe (ML_DESIGN §3.3): can a linear model read camera / CCD / Tmag-bin off the
shape embedding? Fit on train-split stars, report accuracy on val-split stars vs chance
(majority-class rate on the val labels)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from oddball.ml.scoring import tmag_bin


def probe_labels(index_rows) -> dict[str, np.ndarray]:
    return {
        "camera": index_rows.camera.to_numpy(),
        "camera_ccd": (index_rows.camera * 10 + index_rows.ccd).to_numpy(),
        "tmag_bin": tmag_bin(index_rows.tmag.to_numpy()),
    }


def linear_probe(z_tr, y_tr, z_va, y_va, max_iter: int = 300) -> dict:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=max_iter, C=1.0))
    clf.fit(z_tr, y_tr)
    acc = float((clf.predict(z_va) == y_va).mean())
    _, counts = np.unique(y_va, return_counts=True)
    chance = float(counts.max() / counts.sum())
    return {"acc": round(acc, 4), "chance": round(chance, 4), "n_classes": len(counts)}


def run_probes(z_tr, idx_tr, z_va, idx_va) -> dict:
    ltr, lva = probe_labels(idx_tr), probe_labels(idx_va)
    return {k: linear_probe(z_tr, ltr[k], z_va, lva[k]) for k in ltr}
