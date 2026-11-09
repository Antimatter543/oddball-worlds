import numpy as np
import pytest

torch = pytest.importorskip("torch")

from oddball.ml.data import GRID_LEN, PATCH
from oddball.ml.diagnostics import interp_baseline, sinusoid_test, synthetic_sinusoids


def test_interp_baseline_linear_across_hidden_block():
    y = torch.arange(20, dtype=torch.float32)[None, :] * 2.0
    vis = torch.ones(1, 20, dtype=torch.bool)
    vis[0, 5:10] = False
    out = interp_baseline(y, vis)
    assert torch.allclose(out, y)  # a line is interpolated exactly
    vis2 = torch.ones(1, 20, dtype=torch.bool)
    vis2[0, :3] = False  # leading hidden: constant extrapolation from the first visible bin
    vis2[0, -2:] = False
    out2 = interp_baseline(y, vis2)
    assert (out2[0, :3] == y[0, 3]).all() and (out2[0, -2:] == y[0, -3]).all()


def test_synthetic_sinusoids_are_in_shape_units():
    x = synthetic_sinusoids(4, seed=0)
    assert x.shape == (4, 3, GRID_LEN)
    pdc = x[:, 0]
    assert np.isnan(pdc[:, :20]).all() and np.isnan(pdc[:, 600:648]).all()
    s = np.nanstd(pdc, axis=1)
    assert ((s > 0.7) & (s < 1.5)).all()  # robust-sigma units (sinusoid: MAD-sigma ~ 0.7 amp)


class _Oracle(torch.nn.Module):
    """Returns the true (hidden) signal: the sinusoid test must give correlation ~1."""

    def __init__(self, truth):
        super().__init__()
        self.truth = truth

    def forward(self, inp, mask):
        mu = torch.stack([self.truth, self.truth], 1)
        return mu, torch.zeros_like(mu), None


class _Median(torch.nn.Module):
    def forward(self, inp, mask):
        B, P, _ = inp.shape
        mu = torch.zeros(B, 2, P * PATCH)
        return mu, torch.zeros_like(mu), None


def test_sinusoid_test_separates_oracle_from_median():
    x = torch.from_numpy(synthetic_sinusoids(16, seed=0))
    truth = torch.nan_to_num(x[:, 0])
    good = sinusoid_test(_Oracle(truth), torch.device("cpu"), drop_sap=True, n=16)
    bad = sinusoid_test(_Median(), torch.device("cpu"), drop_sap=True, n=16)
    assert good["hide1"]["corr_mean"] > 0.99 and good["hide4"]["corr_pooled"] > 0.99
    assert bad["hide1"]["corr_mean"] == 0.0
    assert bad["hide4"]["mse_zero"] == pytest.approx(bad["hide4"]["mse_model"])
