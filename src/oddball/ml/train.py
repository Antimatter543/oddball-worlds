"""Training loop: AdamW + warmup/cosine, fp32 (optional fp16 autocast), early stopping on val NLL.

Trains on one or more caches (one per sector, ``MultiCache``); each epoch also logs whether the
reconstruction uses context (``oddball.ml.diagnostics``).

Every run writes ``runs/<run_id>/``: config.yaml, git_sha.txt, manifest.json (data provenance,
split counts, exclusion list), metrics.jsonl, best.pt, last.pt and a TensorBoard dir (ML_DESIGN §3).
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from oddball.common.paths import REPO_ROOT
from oddball.common.splits import known_positive_tics
from oddball.ml.data import (
    Cache,
    MultiCache,
    augment,
    contiguous_block_mask,
    make_inputs,
    sector_gap_masks,
    mixed_order,
    signal_dominated,
    signal_flags,
)
from oddball.ml.diagnostics import masked_errors, sinusoid_test, summarise
from oddball.ml.model import ModelConfig, PatchMAE, gaussian_nll, n_params, patch_to_bin


@dataclass
class TrainConfig:
    run_id: str = ""
    cache_dirs: list[str] = field(default_factory=list)
    seed: int = 0
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup_frac: float = 0.03
    max_epochs: int = 12
    patience: int = 3
    mask_ratio: float = 0.4
    amp: bool = False  # fp16 measured 2x slower than fp32 on this GTX 1650S + cu13.2 wheel
    common_gap_mask: bool = False
    drop_sap: bool = False
    val_max: int = 8000
    max_steps_per_epoch: int = 0  # 0 = full epoch; >0 for smoke tests / overfit
    overfit_rows: int = 0  # >0: train on this many fixed train rows with no augmentation
    augment: bool = True
    block_min: int = 2  # masked block length in patches (16 bins = 8 h each)
    block_max: int = 8
    sig_n: int = 1000  # signal-dominated val rows for the per-epoch context check (0 = off)
    sig_frac: float = 0.0  # share of each batch drawn from signal-dominated train rows


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True)
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT).returncode != 0
        return sha.strip() + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _provenance(cache: Cache) -> dict:
    from oddball.common.paths import sector_dir

    out = {}
    for s in cache.info.get("sectors", []):
        p = sector_dir(int(s)) / "provenance.json"
        if p.exists():
            d = json.loads(p.read_text())
            out[str(s)] = {k: d.get(k) for k in ("sector", "product", "product_full",
                                                 "hlsp_version", "doi")}  # fmt: skip
            runs = d.get("runs", [])
            out[str(s)]["download_utc"] = [r.get("started_utc") for r in runs]
    return out


class Batcher:
    """Turns cache batches into model inputs on the device. ``obs_masks`` [n_caches, L] bool
    (per-sector common-gap masks) or None."""

    def __init__(self, device, obs_masks=None, drop_sap=False):
        self.device = device
        self.obs_masks = (None if obs_masks is None
                          else torch.as_tensor(np.asarray(obs_masks), device=device))  # fmt: skip
        self.drop_sap = drop_sap

    def __call__(self, x_np, cid=None, gen=None, aug=False):
        x = torch.from_numpy(x_np).to(self.device, non_blocking=True)
        if self.drop_sap:
            x[:, 1] = float("nan")
        if aug:
            x = augment(x, gen)
        om = None
        if self.obs_masks is not None:
            c = torch.as_tensor(np.zeros(len(x_np), int) if cid is None else cid,
                                device=self.device)  # fmt: skip
            om = self.obs_masks[c]
        return make_inputs(x, om)


def evaluate(model, mc: MultiCache, rows, batcher, cfg: TrainConfig, n_patches: int,
             amp: bool) -> float:  # fmt: skip
    model.eval()
    rng = np.random.default_rng(12345)  # fixed val masks across epochs
    tot, den = 0.0, 0
    with torch.no_grad():
        for _, cid, x, _ in mc.iter_batches(rows, cfg.batch_size, shuffle=False):
            inp, tgt, err, obs, sobs = batcher(x, cid)
            m = torch.from_numpy(contiguous_block_mask(len(x), n_patches, cfg.mask_ratio, rng))
            m = m.to(inp.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                mu, logv, _ = model(inp, m)
            loss = gaussian_nll(mu, logv, tgt, err, obs, sobs, patch_to_bin(m),
                                model.cfg.sap_weight)  # fmt: skip
            tot += float(loss) * len(x)
            den += len(x)
    model.train()
    return tot / max(den, 1)


def context_check(model, mc: MultiCache, rows, batcher, cfg: TrainConfig, n_patches: int) -> dict:
    """Masked-bin MSE vs median / interpolation on signal-dominated val stars + sinusoid test
    (``oddball.ml.diagnostics``). Fixed rows and masks, so epochs are comparable."""
    model.eval()
    rng = np.random.default_rng(777)
    sums = []
    with torch.no_grad():
        for _, cid, x, _ in mc.iter_batches(rows, cfg.batch_size, shuffle=False):
            inp, tgt, err, obs, _ = batcher(x, cid)
            m = torch.from_numpy(contiguous_block_mask(len(x), n_patches, cfg.mask_ratio, rng))
            sums.append(masked_errors(model, inp, tgt, err, obs, m.to(inp.device)))
        out = {"sig": summarise(np.concatenate(sums)) if sums else {},
               "sin": sinusoid_test(model, batcher.device, cfg.drop_sap)}  # fmt: skip
    model.train()
    return out


def pick_signal_rows(mc: MultiCache, rows: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Up to ``n`` signal-dominated rows (``data.signal_dominated``) from ``rows``."""
    rng = np.random.default_rng(seed)
    cand = np.sort(rng.choice(rows, min(len(rows), 20 * n), replace=False))
    keep = []
    for g, _cid, x, _ in mc.iter_batches(cand, 1024, shuffle=False):
        keep.append(g[signal_dominated(x)])
    keep = np.concatenate(keep)
    return np.sort(rng.choice(keep, min(n, len(keep)), replace=False))


def train(cfg: TrainConfig, mcfg: ModelConfig, log=print) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"
    run_dir = REPO_ROOT / "runs" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    mc = MultiCache([Path(d) for d in cfg.cache_dirs])
    tr = mc.training_rows("train")
    va = mc.training_rows("val")
    if cfg.overfit_rows:
        tr = tr[: cfg.overfit_rows]
        va = tr
    elif cfg.val_max and len(va) > cfg.val_max:
        va = np.sort(np.random.default_rng(0).choice(va, cfg.val_max, replace=False))
    sig_rows = pick_signal_rows(mc, mc.training_rows("val"), cfg.sig_n) if cfg.sig_n else None
    obs_masks = sector_gap_masks(mc, tr) if cfg.common_gap_mask else None
    sig_tr = np.zeros(0, dtype=np.int64)
    if cfg.sig_frac > 0:
        flags = np.concatenate([signal_flags(c) for c in mc.caches])
        sig_tr = tr[flags[tr]]
        log(f"oversampling {len(sig_tr)} signal-dominated train rows ({cfg.sig_frac:.0%} of "
            "each batch)")  # fmt: skip

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"train": asdict(cfg), "model": mcfg.to_dict()}, sort_keys=False)
    )
    (run_dir / "git_sha.txt").write_text(git_sha() + "\n")
    caches = []
    for c, o0, o1 in zip(mc.caches, mc.offsets[:-1], mc.offsets[1:], strict=True):
        idx = c.index
        in_tr = np.isin(np.arange(len(c)), tr[(tr >= o0) & (tr < o1)] - o0)
        caches.append({
            "dir": str(c.dir), **{k: c.info[k] for k in ("origin", "n", "sectors")},
            "provenance": _provenance(c),
            "split_counts": idx.split.value_counts().to_dict(),
            "n_train": int(in_tr.sum()),
            "known_tics_in_data": sorted(int(t) for t in idx.tic[idx.split == "known"]),
            "n_val": int((idx.split == "val").sum()), "n_test": int((idx.split == "test").sum()),
            "n_train_dropped_low_obs": int(((idx.split == "train") & ~in_tr).sum()),
        })  # fmt: skip
    manifest = {
        "caches": caches, "n_train": len(tr), "n_val_eval": len(va),
        "n_signal_val_rows": 0 if sig_rows is None else len(sig_rows),
        "excluded_from_training": {
            "splits": "only split=='train' rows are trained on; known/val/test never",
            "known_positive_list_size": len(known_positive_tics()),
            "injection_hosts": "none in train by construction (hosts are val/test only)",
        },
        "common_gap_mask_bins": None if obs_masks is None else obs_masks.sum(1).tolist(),
    }  # fmt: skip
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    model = PatchMAE(mcfg).to(device)
    log(f"run {cfg.run_id}: {n_params(model) / 1e6:.2f}M params, sectors {mc.sectors}, "
        f"train {len(tr)}, val {len(va)}, signal-val {0 if sig_rows is None else len(sig_rows)}, "
        f"amp={amp}, device={device}")  # fmt: skip
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))  # fmt: skip
    steps_per_epoch = math.ceil(len(tr) / (cfg.batch_size - round(cfg.sig_frac * cfg.batch_size)))
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    total = steps_per_epoch * cfg.max_epochs
    warm = max(1, int(cfg.warmup_frac * total))

    def lr_at(step):
        if step < warm:
            return cfg.lr * (step + 1) / warm
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(str(run_dir / "tb"))
    batcher = Batcher(device, obs_masks, cfg.drop_sap)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    mrng = np.random.default_rng(cfg.seed)
    P = mcfg.n_patches
    mf = (run_dir / "metrics.jsonl").open("a")
    best, bad, step = float("inf"), 0, 0
    epoch_rng = np.random.default_rng(cfg.seed + 99)
    for epoch in range(cfg.max_epochs):
        t0 = time.time()
        run_loss, nb = 0.0, 0
        # with max_steps_per_epoch, each "epoch" is a fresh random subset (not the same prefix)
        order = mixed_order(tr, sig_tr, cfg.sig_frac, cfg.batch_size, epoch_rng)
        for i, (_, cid, x, _) in enumerate(mc.iter_batches(order, cfg.batch_size, shuffle=False)):
            if cfg.max_steps_per_epoch and i >= cfg.max_steps_per_epoch:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            inp, tgt, err, obs, sobs = batcher(x, cid, gen,
                                               aug=cfg.augment and not cfg.overfit_rows)
            m = torch.from_numpy(contiguous_block_mask(len(x), P, cfg.mask_ratio, mrng,
                                                       cfg.block_min, cfg.block_max)).to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                mu, logv, _ = model(inp, m)
            loss = gaussian_nll(mu, logv, tgt, err, obs, sobs, patch_to_bin(m), mcfg.sap_weight,
                                beta=mcfg.nll_beta)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            lv = float(loss)
            if not math.isfinite(lv):
                log(f"  non-finite loss at step {step}")
            run_loss += lv
            nb += 1
            if step % 50 == 0:
                tb.add_scalar("train/loss", lv, step)
                tb.add_scalar("train/lr", lr_at(step), step)
                tb.add_scalar("train/grad_norm", float(gn), step)
                mf.write(json.dumps({"step": step, "epoch": epoch, "train_loss": lv,
                                     "lr": lr_at(step), "grad_norm": float(gn)}) + "\n")  # fmt: skip
            step += 1
        t_train = time.time() - t0
        vl = evaluate(model, mc, va, batcher, cfg, P, amp)
        rec = {"epoch": epoch, "step": step, "train_loss_epoch": run_loss / max(nb, 1),
               "val_nll": vl, "epoch_seconds": round(t_train, 1)}  # fmt: skip
        if sig_rows is not None:
            cc = context_check(model, mc, sig_rows, batcher, cfg, P)
            rec["ctx"] = cc
            tb.add_scalar("val/sig_gain_vs_zero", cc["sig"]["gain_vs_zero"], step)
            tb.add_scalar("val/sin_corr_hide1", cc["sin"]["hide1"]["corr_mean"], step)
        rec["eval_seconds"] = round(time.time() - t0 - t_train, 1)
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        tb.add_scalar("val/nll", vl, step)
        tb.add_scalar("train/loss_epoch", rec["train_loss_epoch"], step)
        msg = ""
        if "ctx" in rec:
            sg, sn = rec["ctx"]["sig"], rec["ctx"]["sin"]
            msg = (f" | sig mse {sg['mse_model']:.3f} vs zero {sg['mse_zero']:.3f} interp "
                   f"{sg['mse_interp']:.3f} (gain {sg['gain_vs_zero']:+.2f}) | sin corr "
                   f"h1 {sn['hide1']['corr_mean']:+.2f} h4 {sn['hide4']['corr_mean']:+.2f}")
        log(f"  epoch {epoch}: train {rec['train_loss_epoch']:.4f} val {vl:.4f} "
            f"({t_train:.0f}s + {rec['eval_seconds']:.0f}s){msg}")  # fmt: skip
        ckpt = {"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "train_cfg": asdict(cfg),
                "epoch": epoch, "val_nll": vl, "obs_mask": None,
                "obs_masks": None if obs_masks is None else
                {int(s): obs_masks[k].tolist() for k, s in enumerate(mc.sectors)},
                "sectors": mc.sectors,
                "origin": {int(s): int(c.info["origin"])
                           for s, c in zip(mc.sectors, mc.caches, strict=True)}}  # fmt: skip
        torch.save(ckpt, run_dir / "last.pt")
        if vl < best - 1e-4:
            best, bad = vl, 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"  early stop at epoch {epoch} (best val {best:.4f})")
                break
    mf.close()
    tb.close()
    return run_dir


def load_model(ckpt_path: Path, device=None) -> tuple[PatchMAE, dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ck["model_cfg"])
    cfg.setdefault("pos_mode", "learned")  # checkpoints from before pos_mode existed
    cfg.setdefault("nll_beta", 0.0 if "pos_mode" not in ck["model_cfg"] else 0.5)
    cfg.setdefault("embed_norm", False)
    model = PatchMAE(ModelConfig(**cfg)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
