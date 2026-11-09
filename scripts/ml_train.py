"""Train the self-supervised patch transformer on the ``train`` split of one or more sectors.

    # ssl_v1 (S32 only)
    uv run python -I scripts/ml_train.py --run-id ssl_v1_s0 --sectors 32
    # ssl_v2: five sectors, S32 held out entirely, no SAP channel, normalised patch embedding
    uv run python -I scripts/ml_train.py --run-id ssl_v2_s0 --sectors 33 38 43 44 55 \
        --drop-sap --common-gap-mask --embed-norm --seed 0
"""

from __future__ import annotations

import argparse
import datetime as dt

from oddball.common.paths import sector_dir
from oddball.ml.data import build_cache, cache_dir_for
from oddball.ml.model import ModelConfig
from oddball.ml.train import TrainConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sectors", type=int, nargs="+", default=[32])
    ap.add_argument("--run-id", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    ap.add_argument("--overfit-rows", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (slower on GTX 1650S)")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--common-gap-mask", action="store_true")
    ap.add_argument("--drop-sap", action="store_true")
    ap.add_argument("--arch", default="transformer", choices=["transformer", "cnn", "hybrid"])
    ap.add_argument("--n-conv", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=10)
    ap.add_argument("--nll-beta", type=float, default=0.5)
    ap.add_argument("--pos-mode", default="sincos", choices=["sincos", "learned"])
    ap.add_argument("--embed-norm", action="store_true")
    ap.add_argument("--mask-ratio", type=float, default=0.4)
    ap.add_argument("--block-min", type=int, default=2)
    ap.add_argument("--block-max", type=int, default=8)
    ap.add_argument("--sig-n", type=int, default=1000)
    ap.add_argument("--sig-frac", type=float, default=0.0)
    ap.add_argument("--val-max", type=int, default=8000)
    a = ap.parse_args()

    caches = [str(build_cache(sector_dir(s), cache_dir_for(sector_dir(s)))) for s in a.sectors]
    run_id = a.run_id or dt.datetime.now().strftime("ssl_%Y%m%d_%H%M%S")
    cfg = TrainConfig(
        run_id=run_id, cache_dirs=caches, seed=a.seed, batch_size=a.batch_size, lr=a.lr,
        max_epochs=a.epochs, patience=a.patience, amp=a.amp,
        common_gap_mask=a.common_gap_mask, drop_sap=a.drop_sap,
        max_steps_per_epoch=a.max_steps_per_epoch, overfit_rows=a.overfit_rows,
        augment=not a.no_augment, mask_ratio=a.mask_ratio, block_min=a.block_min,
        block_max=a.block_max, sig_n=a.sig_n, val_max=a.val_max, sig_frac=a.sig_frac,
    )  # fmt: skip
    mcfg = ModelConfig(arch=a.arch, d_model=a.d_model, n_layers=a.layers, nll_beta=a.nll_beta,
                       pos_mode=a.pos_mode, embed_norm=a.embed_norm, n_conv=a.n_conv)  # fmt: skip
    out = train(cfg, mcfg, log=lambda s: print(s, flush=True))
    print(f"run dir: {out}")


if __name__ == "__main__":
    main()
