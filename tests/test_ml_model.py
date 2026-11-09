import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oddball.ml.data import GRID_LEN, PATCH, make_inputs
from oddball.ml.model import ModelConfig, PatchMAE, gaussian_nll, patch_to_bin
from oddball.ml.scoring import knn_score, percentile_vs, stratified_percentile


def _model():
    torch.manual_seed(0)
    return PatchMAE(ModelConfig(d_model=32, n_layers=2, n_heads=4)).eval()


def _x(B=2):
    x = torch.randn(B, 3, GRID_LEN)
    x[:, 2] = 1.0
    return x


def test_forward_shapes_and_embedding():
    m = _model()
    inp, _tgt, _err, _obs, _sobs = make_inputs(_x())
    mask = torch.zeros(2, GRID_LEN // PATCH, dtype=torch.bool)
    mask[:, 10:30] = True
    mu, logv, _h = m(inp, mask)
    assert mu.shape == (2, 2, GRID_LEN) and logv.shape == mu.shape
    assert m.embed_pooled(inp).shape == (2, 32)


def test_masked_patch_hides_every_channel():
    """Changing PDC *or SAP* inside a hidden patch must not change any prediction: the model
    cannot copy the hidden answer from another channel."""
    m = _model()
    x = _x()
    mask = torch.zeros(2, GRID_LEN // PATCH, dtype=torch.bool)
    mask[:, 20:28] = True
    lo, hi = 20 * PATCH, 28 * PATCH
    with torch.no_grad():
        a, _, _ = m(make_inputs(x)[0], mask)
        x2 = x.clone()
        x2[:, 0, lo:hi] += 5.0
        x2[:, 1, lo:hi] -= 5.0
        b, _, _ = m(make_inputs(x2)[0], mask)
    assert torch.allclose(a, b, atol=1e-5)


def test_nll_ignores_unobserved_and_visible_bins():
    m = _model()
    x = _x()
    x[0, :, 0:200] = float("nan")
    inp, tgt, err, obs, sobs = make_inputs(x)
    mask = torch.zeros(2, GRID_LEN // PATCH, dtype=torch.bool)
    mask[:, :20] = True
    with torch.no_grad():
        mu, logv, _ = m(inp, mask)
        l1 = gaussian_nll(mu, logv, tgt, err, obs, sobs, patch_to_bin(mask))
        tgt2 = tgt.clone()
        tgt2[0, :, 0:200] = 99.0  # unobserved bins
        tgt2[:, :, 20 * PATCH :] = -99.0  # visible (not hidden) bins
        l2 = gaussian_nll(mu, logv, tgt2, err, obs, sobs, patch_to_bin(mask))
    assert torch.allclose(l1, l2)


def test_percentiles_and_strata():
    ref = np.arange(1000, dtype=float)
    p = percentile_vs(ref, np.array([-1.0, 499.5, 2000.0, np.nan]))
    assert p[0] == 0 and abs(p[1] - 0.5) < 1e-3 and p[2] == 1 and p[3] == 0.5
    # small fine strata fall back to the coarse stratum
    x = np.array([5.0, 5.0])
    out = stratified_percentile(x, np.array([1, 2]), np.array([0, 0]), np.r_[ref, ref],
                                np.r_[np.full(1000, 1), np.full(1000, 3)], np.zeros(2000, int),
                                min_n=500)  # fmt: skip
    assert np.allclose(out, 5.5 / 1000, atol=1e-3)


def test_knn_excludes_own_tic():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((200, 8)).astype(np.float32)
    tic = np.arange(200)
    s_self = knn_score(ref[:5], tic[:5], ref, tic, k=3, device=torch.device("cpu"))
    s_incl = knn_score(ref[:5], np.full(5, -1), ref, tic, k=3, device=torch.device("cpu"))
    assert (s_self > s_incl).all()  # including self would give a zero distance


def test_masked_tokens_are_position_distinguishable():
    """Regression for runs ssl_v1_s0 / ssl_v1b_s0: with tiny learned positions every hidden
    token looked the same, so the model could only predict one value for all hidden patches.
    With sin-cos positions, hidden tokens at different positions must differ at init."""
    torch.manual_seed(0)
    m = PatchMAE(ModelConfig(d_model=32, n_layers=2, n_heads=4)).eval()
    inp = make_inputs(_x(1))[0]
    mask = torch.zeros(1, GRID_LEN // PATCH, dtype=torch.bool)
    mask[:, 10:50] = True
    with torch.no_grad():
        h = m.tokens(inp, mask)[0, 10:50]
    spread = (h - h.mean(0)).norm(dim=1).mean() / h.norm(dim=1).mean()
    assert spread > 0.3
