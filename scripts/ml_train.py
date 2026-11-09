"""Train the self-supervised patch transformer on the real S32 ``train`` split.

    ODDBALL_DATA=... uv run python -I scripts/ml_train.py --run-id ssl_v1_s0 --seed 0
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
    ap.add_argument("--sector", type=int, default=32)
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
    ap.add_argument("--arch", default="transformer", choices=["transformer", "cnn"])
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=10)
    a = ap.parse_args()

    cache = build_cache(sector_dir(a.sector), cache_dir_for(sector_dir(a.sector)))
    run_id = a.run_id or dt.datetime.now().strftime("ssl_%Y%m%d_%H%M%S")
    cfg = TrainConfig(
        run_id=run_id, cache_dirs=[str(cache)], seed=a.seed, batch_size=a.batch_size, lr=a.lr,
        max_epochs=a.epochs, patience=a.patience, amp=a.amp,
        common_gap_mask=a.common_gap_mask, drop_sap=a.drop_sap,
        max_steps_per_epoch=a.max_steps_per_epoch, overfit_rows=a.overfit_rows,
        augment=not a.no_augment,
    )  # fmt: skip
    mcfg = ModelConfig(arch=a.arch, d_model=a.d_model, n_layers=a.layers)
    out = train(cfg, mcfg, log=lambda s: print(s, flush=True))
    print(f"run dir: {out}")


if __name__ == "__main__":
    main()
