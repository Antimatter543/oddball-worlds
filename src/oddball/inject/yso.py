"""Young-star host injection set (``yso1``; docs/ML_DESIGN.md §15).

The generic sets (``v1``, ``eb1``) draw hosts from the whole sector, so almost none of them is a
young star. Real dippers, though, sit on young, spotted, accreting stars whose own variability is
large and irregular, and tuning on generic hosts chose combiners that lose them (§12-§14). This
set puts the dipper / Boyajian / exocomet / dusty-tail templates on **catalogued young-star
hosts** of one split, and adds a **matched null**: one unchanged copy of every host, which
measures how often the host's own variability already reaches the top 1%.

Host pool (:func:`young_host_pool`), from the same catalogue cone results that ``kc_v1`` uses
for its hard exclusions (``oddball.knownclass.labels``):

* a VSX entry within ``VSX_RADIUS`` (21", one TESS pixel) whose type is a young-star or
  irregular-dimming type (``labels.vsx_excluded``), minus the few non-young types in that list
  (``NON_YOUNG``: R CrB, Boyajian-like, ZZ Ceti), **or** a Gaia DR3 ``vari_classifier_result``
  best class ``YSO`` within ``GAIA_RADIUS`` (3");
* ``split_of(tic)`` is the set's split (never ``known``, never ``train``);
* not a host of any earlier injection set (``exclude_tics``);
* more than ``MIN_SEP_KNOWN`` (120", ~6 TESS pixels) from every known positive, so no known
  positive's light is blended into a host (failure catalogue #4, leakage).

Set layout (:func:`build_host_set`): the usual ``<data>/injections/<name>/sNNNN/<split>/`` files
(``oddball.inject.build``), plus a ``replica`` column. Each host carries ``replicas`` independent
injections (``replica`` 0 .. R-1; kinds cycled with a per-host offset, so every host gets every
kind) and one matched null (``kind = "null_yso"``, ``replica = -1``, ``is_null = True``, depth 0:
the light curve is bit-identical to the real one). Within one replica every host appears once,
so cross-star vetoes can be run per replica without a host's copies seeing each other.
``hosts.parquet`` records each host's catalogue type and match.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa

from oddball.common.paths import sector_dir
from oddball.common.splits import known_positive_tics, split_array
from oddball.common.strata import stratum_id
from oddball.inject.build import (
    SPLIT_CODE,
    _atomic,
    _provenance,
    _write_shards,
    _write_tables,
    injection_dir,
)
from oddball.knownclass.labels import vsx_excluded, vsx_tokens
from oddball.tess.arrays import dir_metadata

YSO_KINDS = ("aperiodic_dipper", "quasiperiodic_dipper", "boyajian", "exocomet", "dusty_tail")
# log-uniform depths spanning the ~0.3 %, ~1 %, ~3 % and ~10 % evaluation bins (harness edges)
YSO_DEPTH_RANGE = (0.00173, 0.173)
NULL_KIND = "null_yso"
NON_YOUNG = {"RCB", "BOY", "ZZO"}
GAIA_YOUNG = {"YSO"}
VSX_RADIUS = 21.0  # arcsec, as the kc_v1 exclusion
GAIA_RADIUS = 3.0  # arcsec
MIN_SEP_KNOWN = 120.0  # arcsec


def vsx_young(type_str: str) -> bool:
    """A kc_v1-excluded VSX type that is a young-star / irregular type (not R CrB etc.)."""
    if not vsx_excluded(type_str):
        return False
    toks = {t for t, _ in vsx_tokens(type_str)}
    return not toks or not toks <= NON_YOUNG


def _sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation (arcsec) between every pair (len(ra1) x len(ra2))."""
    r1, d1 = np.radians(np.asarray(ra1, float))[:, None], np.radians(np.asarray(dec1, float))[:, None]
    r2, d2 = np.radians(np.asarray(ra2, float))[None, :], np.radians(np.asarray(dec2, float))[None, :]
    h = (np.sin((d2 - d1) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600.0


def young_host_pool(meta: pd.DataFrame, vsx: pd.DataFrame | None, gaia: pd.DataFrame | None,
                    splits=("val", "test"), exclude_tics=None,
                    known: pd.DataFrame | None = None,
                    min_sep_known: float = MIN_SEP_KNOWN) -> tuple[pd.DataFrame, dict]:  # fmt: skip
    """Catalogued young-star hosts of ``meta`` (one sector) in ``splits``.

    ``vsx``: tic, Name, Type, sep_arcsec (cone results, any radius); ``gaia``: tic, Class,
    sep_arcsec (VizieR ``I/358/vclassre``). ``known``: ra, dec of the known positives (default:
    the ``known``-split rows of ``meta``). Returns ``(pool, counts)``; ``pool`` has the meta
    columns plus ``split``, ``yso_source`` (vsx / gaia / vsx+gaia), ``yso_type``, ``yso_name``,
    ``yso_sep``, ``sep_known`` (arcsec to the nearest known positive).
    """
    m = meta.copy()
    m["split"] = split_array(m.tic.to_numpy())
    rows = {}
    if vsx is not None and len(vsx):
        v = vsx[(vsx.sep_arcsec <= VSX_RADIUS) & vsx.tic.isin(m.tic)].copy()
        v = v[v.Type.map(vsx_young)].sort_values("sep_arcsec").drop_duplicates("tic")
        for r in v.itertuples():
            rows[int(r.tic)] = {"yso_source": "vsx", "yso_type": str(r.Type),
                                "yso_name": str(r.Name), "yso_sep": float(r.sep_arcsec)}
    if gaia is not None and len(gaia):
        g = gaia[(gaia.sep_arcsec <= GAIA_RADIUS) & gaia.tic.isin(m.tic)
                 & gaia.Class.isin(GAIA_YOUNG)]  # fmt: skip
        g = g.sort_values("sep_arcsec").drop_duplicates("tic")
        for r in g.itertuples():
            t = int(r.tic)
            if t in rows:
                rows[t]["yso_source"] = "vsx+gaia"
            else:
                rows[t] = {"yso_source": "gaia", "yso_type": f"Gaia {r.Class}",
                           "yso_name": str(r.Source), "yso_sep": float(r.sep_arcsec)}
    cat = pd.DataFrame.from_dict(rows, orient="index").rename_axis("tic").reset_index()
    if cat.empty:
        cat = pd.DataFrame(columns=["tic", "yso_source", "yso_type", "yso_name", "yso_sep"])
    pool = m.merge(cat, on="tic", how="inner")
    counts = {"catalogued_young": len(pool),
              "by_split": pool.split.value_counts().to_dict()}
    kn = known if known is not None else m[m.split == "known"]
    if len(kn) and len(pool):
        pool["sep_known"] = _sep_arcsec(pool.ra, pool.dec, kn.ra, kn.dec).min(1)
    else:
        pool["sep_known"] = np.inf
    pool = pool[pool.split.isin(splits)]
    counts["in_splits"] = len(pool)
    kt = known_positive_tics()
    assert not pool.tic.isin(kt).any(), "known positives leaked into the young-host pool"
    near = pool.sep_known <= min_sep_known
    counts["near_known"] = int(near.sum())
    pool = pool[~near]
    if exclude_tics is not None:
        used = pool.tic.isin({int(x) for x in exclude_tics})
        counts["earlier_hosts"] = int(used.sum())
        pool = pool[~used]
    counts["pool"] = len(pool)
    counts["pool_by_split"] = pool.split.value_counts().to_dict()
    return pool.sort_values("tic").reset_index(drop=True), counts


def null_params(t: np.ndarray) -> dict:
    """A depth-0 model: the injected light curve equals the host's."""
    tc = float(np.median(t))
    return {"kind": NULL_KIND, "depth": 0.0, "duration_d": 0.0, "sign": -1, "norm": 1.0,
            "t0_btjd": tc, "components": [{"shape": "gauss", "tc": tc, "fwhm": 0.1}],
            "is_null": True}  # fmt: skip


def plan_rows(hosts: pd.DataFrame, replicas: int, kinds=YSO_KINDS) -> pd.DataFrame:
    """One row per (host, replica) plus one matched null per host (``replica = -1``).

    Kind of replica ``r`` of host ``i``: ``kinds[(i + r) % len(kinds)]``, so with
    ``replicas >= len(kinds)`` every host carries every kind.
    """
    parts = []
    for r in range(replicas):
        k = [kinds[(i + r) % len(kinds)] for i in range(len(hosts))]
        parts.append(hosts.assign(kind=k, replica=r, params=None))
    parts.append(hosts.assign(kind=NULL_KIND, replica=-1, params=None))
    return pd.concat(parts, ignore_index=True)


def build_host_set(sector: int, split: str, hosts: pd.DataFrame, name: str = "yso1",
                   seed: int = 20261111, replicas: int = 20, kinds=YSO_KINDS,
                   depth_range=YSO_DEPTH_RANGE, out_dir=None, log=print) -> dict:  # fmt: skip
    """Write the injection set for ``hosts`` (rows of :func:`young_host_pool`, one split)."""
    if split not in ("val", "test"):
        raise ValueError("injection sets are built for val or test only")
    if not (hosts.split == split).all():
        raise ValueError("every host must belong to the set's split")
    out = out_dir or injection_dir(name, sector, split)
    if (out / "injections.parquet").exists():
        raise SystemExit(f"{out} already exists; delete it to rebuild")
    (out / "shards").mkdir(parents=True, exist_ok=True)
    meta = dir_metadata(sector_dir(sector))
    bin_days = float(meta.bin_minutes.iloc[0]) / 1440.0
    h = meta.merge(hosts[["tic"]], on="tic", how="inner").sort_values("tic")
    h["stratum"] = stratum_id(h.tmag, h.camera, h.ccd)
    plan = plan_rows(h.reset_index(drop=True), replicas, kinds)
    plan["event_id"] = -1
    plan["inj_id"] = [f"{name}-s{sector:04d}-{split}-{i:05d}" for i in range(len(plan))]
    plan["inj_index"] = np.arange(len(plan))
    # the matched null's (depth 0) params need the host's time grid
    t_by = {int(r.tic): (int(r.bin0), int(r.n_bins)) for r in h.itertuples()}
    plan["params"] = pd.Series(
        [null_params((t_by[int(t)][0] + np.arange(t_by[int(t)][1]) + 0.5) * bin_days)
         if k == NULL_KIND else None for t, k in zip(plan.tic, plan.kind, strict=True)],
        index=plan.index, dtype=object)  # fmt: skip
    assert not plan[plan.replica >= 0].duplicated(["tic", "replica"]).any()
    inj_key = [seed, sector, SPLIT_CODE[split], 11]
    inj, n = _write_shards(sector, split, plan, meta, out, bin_days, inj_key, log,
                           depth_range=depth_range, extra_cols=("replica",))  # fmt: skip
    _write_tables(inj, out)
    hcols = [c for c in ("tic", "split", "tmag", "camera", "ccd", "ra", "dec", "yso_source",
                         "yso_type", "yso_name", "yso_sep", "sep_known") if c in hosts]
    _atomic_df(hosts[hcols], out / "hosts.parquet")
    prov = _provenance(sector, split, name, seed, inj, 0)
    prov["host_rule"] = ("catalogued young-star hosts (oddball.inject.yso.young_host_pool): "
                         f"VSX young type within {VSX_RADIUS}\" or Gaia DR3 YSO within "
                         f"{GAIA_RADIUS}\"; split_of(tic)==split; > {MIN_SEP_KNOWN}\" from every "
                         "known positive; not an earlier injection host")
    prov.update({"n_hosts": len(h), "replicas": replicas, "kinds_cycle": list(kinds),
                 "depth_range": list(depth_range), "null_kind": NULL_KIND,
                 "stream": "per-injection rng [seed, sector, split code, 11, inj_index]"})
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    log(f"wrote {n} light curves ({len(h)} hosts x {replicas} + nulls) to {out}")
    return prov


def _atomic_df(df: pd.DataFrame, path) -> None:
    _atomic(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), path)
