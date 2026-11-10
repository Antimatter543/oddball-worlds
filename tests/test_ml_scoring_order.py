"""``ssl_v2_recon`` must not depend on row order (ML_DESIGN §15 caveat, fixed 2026-11-10).

The reconstruction surprise hides patches with random block masks. Before the fix the masks
came from one generator consumed in row order, so a light curve's score changed with its row.
Now each light curve's masks are seeded by (run seed, host TIC)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from oddball.ml.data import CTX_NAMES, GRID_LEN
from oddball.ml.model import ModelConfig, PatchMAE
from oddball.ml.scoring import encode_and_surprise


class _FakeCache:
    """Duck-typed :class:`oddball.ml.data.Cache` (x [N,3,L] float16, index, ctx)."""

    def __init__(self, x: np.ndarray, index: pd.DataFrame):
        self.x = x
        self.index = index

    def __len__(self) -> int:
        return len(self.x)

    def ctx(self, rows):
        return self.index.loc[rows, list(CTX_NAMES)].to_numpy(np.float32)


def _cache(n: int = 12, seed: int = 0) -> _FakeCache:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 3, GRID_LEN)).astype(np.float32)
    x[:, 2] = 0.5  # per-bin error
    x[:, :, 100:140] = np.nan  # a gap
    # row n-1 is an unchanged copy of row 2 (a matched null, same host)
    x[n - 1] = x[2]
    tic = 1000 + np.arange(n)
    host = tic.copy()
    host[n - 1] = host[2]
    idx = pd.DataFrame({"tic": tic, "host_tic": host, "log_scatter": -2.0, "log_p2p": -2.0,
                        "tmag": 10.0})  # fmt: skip
    return _FakeCache(x.astype(np.float16), idx)


def _model():
    torch.manual_seed(0)
    return PatchMAE(ModelConfig(d_model=32, n_layers=2, n_heads=4)).eval()


def _score(model, cache, rows, batch_size, mask_seeding="tic"):
    return encode_and_surprise(model, cache, rows, n_masks=5, batch_size=batch_size, seed=0,
                               amp=False, log=lambda s: None, block=8,
                               mask_seeding=mask_seeding)  # fmt: skip


def test_recon_score_is_row_order_invariant():
    model, cache = _model(), _cache()
    n = len(cache)
    rows = np.arange(n)
    base = _score(model, cache, rows, batch_size=5)
    perm = np.random.default_rng(1).permutation(n)
    other = _score(model, cache, perm, batch_size=3)  # other order, other batch composition
    for a, b in zip(base, other, strict=True):
        np.testing.assert_allclose(a[perm], b, rtol=1e-5, atol=1e-5)
    # the matched null copy scores exactly like its host, at another row
    np.testing.assert_allclose(base[1][n - 1], base[1][2], rtol=1e-6, atol=1e-6)
    # a different seed gives different masks (the score is not mask-free)
    alt = encode_and_surprise(model, cache, rows, n_masks=5, batch_size=5, seed=7, amp=False,
                              log=lambda s: None, block=8)  # fmt: skip
    assert not np.allclose(alt[1], base[1])


def test_surprise_out_follows_rows_order():
    model, cache = _model(), _cache()
    n = len(cache)
    perm = np.random.default_rng(2).permutation(n)
    s1 = np.zeros((n, GRID_LEN), np.float32)
    s2 = np.zeros((n, GRID_LEN), np.float32)
    encode_and_surprise(model, cache, np.arange(n), batch_size=4, amp=False, log=lambda s: None,
                        block=8, surprise_out=s1)  # fmt: skip
    encode_and_surprise(model, cache, perm, batch_size=4, amp=False, log=lambda s: None, block=8,
                        surprise_out=s2)  # fmt: skip
    np.testing.assert_allclose(s1[perm], s2, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_stream_seeding_is_row_order_dependent():
    """The old behaviour, kept for reproduction, really did depend on the row (documents the bug)."""
    model, cache = _model(), _cache()
    n = len(cache)
    old = _score(model, cache, np.arange(n), batch_size=5, mask_seeding="stream")
    assert abs(old[1][n - 1] - old[1][2]) > 1e-4


def test_rows_must_be_unique():
    model, cache = _model(), _cache()
    with pytest.raises(ValueError):
        _score(model, cache, np.array([0, 1, 1]), batch_size=4)
