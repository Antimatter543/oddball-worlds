"""Training loop: AdamW + warmup/cosine, fp16 autocast + GradScaler, early stopping on val NLL.

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
    augment,
    common_gap_masks,
    contiguous_block_mask,
    iter_batches,
    make_inputs,
    training_rows,
)
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
    """Turns cache batches into model inputs on the device."""

    def __init__(self, device, obs_mask=None, drop_sap=False):
        self.device = device
        self.obs_mask = None if obs_mask is None else torch.as_tensor(obs_mask, device=device)
        self.drop_sap = drop_sap

    def __call__(self, x_np, gen=None, aug=False):
        x = torch.from_numpy(x_np).to(self.device, non_blocking=True)
        if self.drop_sap:
            x[:, 1] = float("nan")
        if aug:
            x = augment(x, gen)
        return make_inputs(x, self.obs_mask)


def evaluate(model, cache, rows, batcher, cfg: TrainConfig, n_patches: int, amp: bool) -> float:
    model.eval()
    rng = np.random.default_rng(12345)  # fixed val masks across epochs
    tot, den = 0.0, 0
    with torch.no_grad():
        for _, x, _ in iter_batches(cache, rows, cfg.batch_size, shuffle=False):
            inp, tgt, err, obs, sobs = batcher(x)
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


def train(cfg: TrainConfig, mcfg: ModelConfig, log=print) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.amp and device.type == "cuda"
    run_dir = REPO_ROOT / "runs" / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    assert len(cfg.cache_dirs) == 1, "multi-sector training not wired yet"
    cache = Cache(Path(cfg.cache_dirs[0]))
    tr = training_rows(cache, "train")
    va = training_rows(cache, "val")
    if cfg.overfit_rows:
        tr = tr[: cfg.overfit_rows]
        va = tr
    elif cfg.val_max and len(va) > cfg.val_max:
        va = np.sort(np.random.default_rng(0).choice(va, cfg.val_max, replace=False))
    excluded = cache.index[~cache.index.index.isin(tr)]
    obs_mask = common_gap_masks(cache, tr)[0] if cfg.common_gap_mask else None

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"train": asdict(cfg), "model": mcfg.to_dict()}, sort_keys=False)
    )
    (run_dir / "git_sha.txt").write_text(git_sha() + "\n")
    manifest = {
        "caches": [{"dir": str(cache.dir), **{k: cache.info[k] for k in ("origin", "n", "sectors")}}],
        "provenance": _provenance(cache),
        "n_train": len(tr), "n_val_eval": len(va),
        "split_counts": cache.index.split.value_counts().to_dict(),
        "excluded_from_training": {
            "known_tics_in_data": sorted(int(t) for t in cache.index.tic[cache.index.split == "known"]),
            "known_positive_list_size": len(known_positive_tics()),
            "n_val": int((excluded.split == "val").sum()),
            "n_test": int((excluded.split == "test").sum()),
            "n_train_dropped_low_obs": int(((cache.index.split == "train")
                                            & ~cache.index.index.isin(tr)).sum()),
            "injection_hosts": "none in train by construction (hosts are val/test only)",
        },
        "common_gap_mask_bins": None if obs_mask is None else int(obs_mask.sum()),
    }  # fmt: skip
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))

    model = PatchMAE(mcfg).to(device)
    log(f"run {cfg.run_id}: {n_params(model) / 1e6:.2f}M params, train {len(tr)}, val {len(va)}, "
        f"amp={amp}, device={device}")  # fmt: skip
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(0.9, 0.95))  # fmt: skip
    steps_per_epoch = math.ceil(len(tr) / cfg.batch_size)
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
    batcher = Batcher(device, obs_mask, cfg.drop_sap)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)
    mrng = np.random.default_rng(cfg.seed)
    P = mcfg.n_patches
    mf = (run_dir / "metrics.jsonl").open("a")
    best, bad, step = float("inf"), 0, 0
    for epoch in range(cfg.max_epochs):
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for i, (_, x, _) in enumerate(iter_batches(cache, tr, cfg.batch_size, shuffle=True,
                                                   seed=cfg.seed * 1000 + epoch)):  # fmt: skip
            if cfg.max_steps_per_epoch and i >= cfg.max_steps_per_epoch:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            inp, tgt, err, obs, sobs = batcher(x, gen, aug=cfg.augment and not cfg.overfit_rows)
            m = torch.from_numpy(contiguous_block_mask(len(x), P, cfg.mask_ratio, mrng)).to(device)
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
        vl = evaluate(model, cache, va, batcher, cfg, P, amp)
        rec = {"epoch": epoch, "step": step, "train_loss_epoch": run_loss / max(nb, 1),
               "val_nll": vl, "epoch_seconds": round(t_train, 1),
               "eval_seconds": round(time.time() - t0 - t_train, 1)}  # fmt: skip
        mf.write(json.dumps(rec) + "\n")
        mf.flush()
        tb.add_scalar("val/nll", vl, step)
        tb.add_scalar("train/loss_epoch", rec["train_loss_epoch"], step)
        log(f"  epoch {epoch}: train {rec['train_loss_epoch']:.4f} val {vl:.4f} "
            f"({t_train:.0f}s train + {rec['eval_seconds']:.0f}s eval)")  # fmt: skip
        ckpt = {"model": model.state_dict(), "model_cfg": mcfg.to_dict(), "train_cfg": asdict(cfg),
                "epoch": epoch, "val_nll": vl,
                "obs_mask": None if obs_mask is None else obs_mask.tolist(),
                "origin": cache.info["origin"]}  # fmt: skip
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
    model = PatchMAE(ModelConfig(**cfg)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck
