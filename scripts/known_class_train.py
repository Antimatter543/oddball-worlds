"""Train the known-class classifier and predict class probabilities for a sector + injections.

    uv run --group ml python -I scripts/known_class_train.py --name kc_v1 \
        --features kcf_v1 --sectors 32 33 38 43 44 55 --predict 32 --inj v1

Inputs: ``<data>/knownclass/labels.parquet`` (scripts/build_known_class_labels.py) and one or
more feature tables ``outputs/knownclass/features/<table>/sNNNN/{real,inj_<name>}.parquet``
(joined on ``tic, sector, inj_id``; any per-light-curve table plugs in, e.g. ML embeddings).

Fits on the ``train`` split of every listed sector, early-stops and temperature-calibrates on
``val``. ``test`` and ``known`` are never touched. Writes ``outputs/knownclass/<name>/``:
``model.pkl``, ``metrics.json``, ``val_per_class.csv``, ``val_confusion.csv``,
``targets_by_class.csv`` and ``sNNNN/{real,inj_<name>}.parquet`` (``kc_*`` columns).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oddball.common.paths import REPO_ROOT, data_root, outputs_root
from oddball.common.splits import split_array
from oddball.knownclass import model as M
from oddball.knownclass.features import feature_columns, join_tables


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_table(tables: list[str], sector: int, part: str) -> pd.DataFrame:
    frames = []
    for t in tables:
        p = outputs_root() / "knownclass" / "features" / t / f"s{sector:04d}" / f"{part}.parquet"
        frames.append(pd.read_parquet(p))
    return join_tables(frames) if len(frames) > 1 else frames[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="kc_v1")
    ap.add_argument("--features", nargs="+", default=["kcf_v1"])
    ap.add_argument("--sectors", type=int, nargs="+", default=[32, 33, 38, 43, 44, 55])
    ap.add_argument("--predict", type=int, nargs="*", default=[32])
    ap.add_argument("--inj", nargs="*", default=["v1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--other", action="store_true",
                    help="add the 'other' catch-all class (ablation; default is a closed set of "
                         "known classes guarded by typicality)")
    a = ap.parse_args()
    out = outputs_root() / "knownclass" / a.name
    out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(data_root() / "knownclass" / "labels.parquet")
    rows, all_targets = [], []
    for s in a.sectors:
        f = load_table(a.features, s, "real")
        f["split"] = split_array(f.tic.to_numpy())
        f["y"] = M.training_targets(f, labels, other=a.other).to_numpy()
        all_targets.append(f[["tic", "sector", "split", "y"]])
        rows.append(f[(f.y != "") & f.split.isin(["train", "val"])])
        log(f"s{s}: {len(f)} light curves, {int((f.y != '').sum())} targets")
        del f
    tg = pd.concat(all_targets, ignore_index=True)
    data = pd.concat(rows, ignore_index=True)
    del rows
    cols = [c for c in feature_columns(data) if c not in ("split", "y")]

    tr = data[data.split == "train"].copy()
    va = data[data.split == "val"].copy()
    tr["y"] = M.cap_quiet(tr.y, seed=a.seed).to_numpy()
    va_cap = va.copy()
    va_cap["y"] = M.cap_quiet(va.y, seed=a.seed + 1).to_numpy()
    tr, va_cap = tr[tr.y != ""], va_cap[va_cap.y != ""]
    log(f"train {len(tr)} rows, val {len(va_cap)} (quiet capped), {len(cols)} features")

    os.environ.setdefault("OMP_NUM_THREADS", "6")
    t0 = time.time()
    model = M.KnownClassModel(feature_cols=cols).fit(tr, tr.y.to_numpy(), va_cap,
                                                     va_cap.y.to_numpy(), seed=a.seed)
    log(f"fit in {time.time() - t0:.0f} s, {model.clf.n_iter_} iterations, "
        f"T = {model.temperature:.3f}")
    t0 = time.time()
    model.fit_typicality(tr, tr.y.to_numpy(), tr.tic.to_numpy(), seed=a.seed)
    log(f"typicality reference in {time.time() - t0:.0f} s: "
        + ", ".join(f"{c} {len(v['ref'])}" for c, v in model.typ.items()))

    # validation report on every val target (quiet not capped: per-class recall is unaffected)
    pv = model.predict_frame(va, tic=va.tic.to_numpy())
    rep = M.per_class_report(va.y.to_numpy(), pv.kc_pred.to_numpy(), model.classes)
    rep_cap = M.per_class_report(va_cap.y.to_numpy(), model.predict_frame(va_cap, tic=va_cap.tic.to_numpy()).kc_pred.to_numpy(),
                                 model.classes)  # fmt: skip
    conf = M.confusion(va.y.to_numpy(), pv.kc_pred.to_numpy(), model.classes)
    typ_rows = []
    for c in model.typ:
        m = va.y.to_numpy() == c
        typ_rows.append({"class": c, "val members": int(m.sum()),
                         "median t (members)": float(np.median(pv[f"kc_t_{c}"][m])) if m.any()
                         else np.nan,
                         "median t (other/quiet)": float(np.median(
                             pv[f"kc_t_{c}"][np.isin(va.y, ["other", "quiet"])])),
                         "mean explained (members)": float(pv.kc_explained[m].mean()) if m.any()
                         else np.nan})  # fmt: skip
    typ = pd.DataFrame(typ_rows)
    typ.to_csv(out / "val_typicality.csv", index=False)
    print(typ.to_string(index=False))
    rep.to_csv(out / "val_per_class.csv", index=False)
    conf.to_csv(out / "val_confusion.csv")
    print(rep.to_string(index=False))
    print(conf.to_string())

    # target counts: class x split (light curves), plus labelled TICs by class x source
    tcount = tg[tg.y != ""].groupby(["y", "split"]).size().unstack(fill_value=0)
    tcount.to_csv(out / "targets_by_class.csv")
    print(tcount.to_string())

    with (out / "model.pkl").open("wb") as fh:
        pickle.dump(model, fh)

    for s in a.predict:
        d = out / f"s{s:04d}"
        d.mkdir(parents=True, exist_ok=True)
        parts = ["real"] + [f"inj_{n}" for n in a.inj]
        for part in parts:
            f = load_table(a.features, s, part)
            p = model.predict_frame(f, tic=f.tic.to_numpy())
            res = pd.concat([f[["tic", "sector", "inj_id"]], p], axis=1)
            if part == "real":
                lab = labels.set_index("tic")
                res["cat_label"] = lab.label.reindex(f.tic).fillna("").to_numpy()
                res["cat_excluded"] = lab.excluded.reindex(f.tic).fillna(False).to_numpy()
                res["target"] = M.training_targets(f, labels, other=a.other).to_numpy()
            res.to_parquet(d / f"{part}.parquet", index=False)
            log(f"predicted s{s} {part}: {len(res)} rows; "
                f"pred mix {res.kc_pred.value_counts(normalize=True).round(3).to_dict()}")

    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=False).stdout.strip()  # fmt: skip
    metrics = {
        "name": a.name, "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git_sha": sha, "features": a.features, "feature_cols": cols, "sectors": a.sectors,
        "classes": model.classes, "temperature": model.temperature,
        "n_iter": int(model.clf.n_iter_), "params": model.params, "other_class": a.other,
        "class_weights": model.class_weights,
        "n_train": len(tr), "n_val_capped": len(va_cap), "n_val": len(va),
        "val_accuracy": float(rep.recall.iloc[-1]),
        "val_accuracy_quiet_capped": float(rep_cap.recall.iloc[-1]),
        "val_per_class": rep.to_dict("records"),
        "gates": {"quiet_max_var_frac": M.QUIET_MAX_VAR_FRAC, "quiet_max_snr": M.QUIET_MAX_SNR,
                  "var_min_var_frac": M.VAR_MIN_VAR_FRAC, "var_min_dip_snr": M.VAR_MIN_DIP_SNR,
                  "quiet_cap": M.QUIET_PER_CLASS_CAP},
        "labels_provenance": str(data_root() / "knownclass" / "labels_provenance.json"),
    }  # fmt: skip
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
