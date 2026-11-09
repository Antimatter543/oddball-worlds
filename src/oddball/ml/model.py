"""PatchTST-style masked-patch transformer with a per-bin Gaussian NLL (ML_DESIGN §2).

* Tokens: one per 16-bin patch; the patch's (pdc, sap, log err, obs) values -> linear -> d_model.
* Masked patches: the whole token (all channels, incl. SAP and the mask channel) is replaced by a
  learned mask token, so the model cannot copy a hidden PDC value from the visible SAP channel.
* Head: per bin, mean and log excess variance for pdc and sap. ``sigma^2 = err^2 + exp(logv)``:
  the measured per-bin error is a floor, so noisy bins count less; the model can only *add*
  variance (to absorb unpredictable stellar variability).
* Loss: NLL over observed bins inside masked patches only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from oddball.ml.data import GRID_LEN, PATCH


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 10
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    n_patches: int = GRID_LEN // PATCH
    patch: int = PATCH
    in_ch: int = 4
    sap_weight: float = 0.5
    # beta-NLL (Seitzer et al. 2022): per-bin NLL x stopgrad(var)^beta. With beta=0 the mean's
    # gradient is scaled by 1/var and the model collapses to "predict the median, inflate the
    # variance" (observed in run ssl_v1_s0, beta=0). 0.5 keeps noisy bins down-weighted.
    nll_beta: float = 0.5
    # "sincos": fixed unit-scale sin-cos positions (MAE). "learned" (std-0.02 init, as in runs
    # ssl_v1_s0 / ssl_v1b_s0) left every masked token ~identical, so the model predicted the same
    # value (the median) at every hidden position and never used context.
    pos_mode: str = "sincos"
    # LayerNorm on the patch (content) embedding before positions are added. Without it content
    # tokens are ~25x larger than the unit-scale sin-cos positions, so attention from a hidden
    # (position-only) token cannot find its neighbours by position (ssl_v2, ML_DESIGN §10).
    embed_norm: bool = False
    arch: str = "transformer"  # or "cnn" (fallback), or "hybrid" (conv stem + transformer)
    n_conv: int = 3  # conv stem blocks for arch="hybrid"

    def to_dict(self) -> dict:
        return asdict(self)


class PatchMAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.embed = nn.Linear(cfg.patch * cfg.in_ch, d)
        if cfg.embed_norm:
            self.embed_ln = nn.LayerNorm(d)
        self.mask_token = nn.Parameter(torch.zeros(d))
        self.pos = nn.Parameter(torch.zeros(1, cfg.n_patches, d))
        nn.init.normal_(self.pos, std=0.02)
        self.register_buffer("pos_fixed", _sincos(cfg.n_patches, d), persistent=False)
        nn.init.normal_(self.mask_token, std=0.02)
        if cfg.arch in ("transformer", "hybrid"):
            layer = nn.TransformerEncoderLayer(
                d, cfg.n_heads, d * cfg.ff_mult, cfg.dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )  # fmt: skip
            tf = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
            if cfg.arch == "hybrid":
                # residual dilated-conv stem (receptive field +-2*(1+2+4)=14 patches): a hidden
                # token gets its neighbours' content before any attention has to be learned
                stem = [_ConvBlock(d, 2**i) for i in range(cfg.n_conv)]
                self.encoder = nn.Sequential(_Transpose(nn.Sequential(*stem)), tf)
            else:
                self.encoder = tf
        elif cfg.arch == "cnn":
            blocks = []
            for i in range(cfg.n_layers):
                blocks += [nn.Conv1d(d, d, 5, padding=2 * (1 + i % 3), dilation=1 + i % 3),
                           nn.GroupNorm(8, d), nn.GELU()]  # fmt: skip
            self.encoder = _CNN(nn.Sequential(*blocks))
        else:
            raise ValueError(cfg.arch)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.patch * 4)

    def tokens(self, inp: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """inp [B,P,patch*4] -> encoder output [B,P,d]. ``mask`` [B,P] bool hides patches."""
        h = self.embed(inp)
        if self.cfg.embed_norm:
            h = self.embed_ln(h)
        if mask is not None:
            h = torch.where(mask[..., None], self.mask_token.to(h.dtype).expand_as(h), h)
        h = h + self.pos
        if self.cfg.pos_mode == "sincos":
            h = h + self.pos_fixed
        return self.norm(self.encoder(h))

    def forward(self, inp, mask=None):
        h = self.tokens(inp, mask)
        B, P, _ = h.shape
        out = self.head(h).view(B, P, self.cfg.patch, 4).permute(0, 3, 1, 2)  # [B,4,P,patch]
        out = out.reshape(B, 4, P * self.cfg.patch)
        mu, logv = out[:, :2], out[:, 2:]
        return mu, logv, h

    @torch.no_grad()
    def embed_pooled(self, inp: torch.Tensor) -> torch.Tensor:
        """Mean over patches that contain any observed bin (no masking at inference)."""
        h = self.tokens(inp, None).float()
        has = (inp[..., 3 :: self.cfg.in_ch].sum(-1) > 0).float()  # obs channel per patch
        has = torch.where(has.sum(1, keepdim=True) > 0, has, torch.ones_like(has))
        return (h * has[..., None]).sum(1) / has.sum(1, keepdim=True)


def _sincos(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    i = torch.arange(d // 2, dtype=torch.float32)[None, :]
    ang = pos / (10000 ** (2 * i / d))
    return torch.cat([torch.sin(ang), torch.cos(ang)], 1)[None]


class _ConvBlock(nn.Module):
    """Pre-norm residual block on [B,d,P]: GroupNorm -> dilated conv (k=5) -> GELU -> 1x1."""

    def __init__(self, d: int, dilation: int):
        super().__init__()
        self.f = nn.Sequential(nn.GroupNorm(8, d), nn.Conv1d(d, d, 5, padding=2 * dilation,
                                                              dilation=dilation),
                               nn.GELU(), nn.Conv1d(d, d, 1))  # fmt: skip

    def forward(self, h):
        return h + self.f(h)


class _Transpose(nn.Module):
    """Apply a [B,d,P] module to [B,P,d] tokens."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, h):
        return self.m(h.transpose(1, 2)).transpose(1, 2)


class _CNN(nn.Module):
    def __init__(self, seq):
        super().__init__()
        self.seq = seq

    def forward(self, h):
        return h + self.seq(h.transpose(1, 2)).transpose(1, 2)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def gaussian_nll(mu, logv, target, err, obs, sap_obs, bin_mask, sap_weight=0.5, reduce=True,
                 beta=0.0):
    """Per-bin Gaussian NLL (constant dropped) on observed bins where ``bin_mask`` [B,L] is True.

    Returns the scalar mean loss (``reduce``) or the per-bin pdc and sap NLL tensors."""
    mu, logv, target = mu.float(), logv.float(), target.float()
    var = err.float()[:, None, :] ** 2 + F.softplus(logv) + 1e-4
    nll = 0.5 * ((target - mu) ** 2 / var + torch.log(var))
    if beta and reduce:
        nll = nll * var.detach() ** beta
    w_pdc = (obs & bin_mask).float()
    w_sap = (sap_obs & bin_mask).float()
    if not reduce:
        return nll, var
    num = (nll[:, 0] * w_pdc).sum() + sap_weight * (nll[:, 1] * w_sap).sum()
    den = w_pdc.sum() + sap_weight * w_sap.sum()
    return num / den.clamp(min=1.0)


def patch_to_bin(mask: torch.Tensor, patch: int = PATCH) -> torch.Tensor:
    return mask.repeat_interleave(patch, dim=1)
