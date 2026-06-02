"""OASIS Stage-1 v3: differentiable, downstream-objective PyTorch model + ops.

All marginals are equi-depth quantile boundaries on normalized [0,1]:
  boundaries_x = [0, q_1, ..., q_{B-1}, 1]  (strictly increasing)
  level(i)     = i / B
Selectivity of `x <= v` is the piecewise-linear CDF at v. Everything here is
differentiable w.r.t. the model-produced boundaries.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PREDICATE_ORDER = ["<", "<=", ">", ">=", "BETWEEN", "="]
N_PRED = len(PREDICATE_ORDER)
OBS_FEAT_DIM = N_PRED + 6  # 6-hot + value, value_upper, est_sel, act_sel, residual, has_upper
EPS = 1e-6


# ───────────────────────── differentiable marginal ops ─────────────────────────

def boundaries_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """(.., B) width logits -> (.., B+1) strictly-increasing x in [0,1]."""
    widths = F.softmax(logits, dim=-1)
    widths = widths + EPS
    widths = widths / widths.sum(dim=-1, keepdim=True)
    cum = torch.cumsum(widths, dim=-1)
    zero = torch.zeros_like(cum[..., :1])
    return torch.cat([zero, cum], dim=-1)  # (.., B+1), starts 0 ends 1


def cdf_at(boundaries_x: torch.Tensor, levels: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Piecewise-linear CDF. boundaries_x:(...,P) levels:(P,) x:(...,) -> (...,).
    Batched over leading dims that match boundaries_x[..., 0]."""
    P = boundaries_x.shape[-1]
    xc = x.clamp(0.0, 1.0).unsqueeze(-1)  # (...,1)
    bx = boundaries_x  # (...,P)
    # find interval: count of boundaries <= x
    idx = (bx <= xc).sum(dim=-1).clamp(1, P - 1)  # (...,)
    g = idx.unsqueeze(-1)
    x0 = torch.gather(bx, -1, g - 1).squeeze(-1)
    x1 = torch.gather(bx, -1, g).squeeze(-1)
    lev = levels.to(boundaries_x.device).expand_as(bx)
    p0 = torch.gather(lev, -1, g - 1).squeeze(-1)
    p1 = torch.gather(lev, -1, g).squeeze(-1)
    denom = (x1 - x0).clamp_min(EPS)
    frac = ((x.clamp(0.0, 1.0) - x0) / denom).clamp(0.0, 1.0)
    out = p0 + frac * (p1 - p0)
    out = torch.where(x <= bx[..., 0], lev[..., 0], out)
    out = torch.where(x >= bx[..., -1], lev[..., -1], out)
    return out.clamp(0.0, 1.0)


def cell_masses(boundaries_x: torch.Tensor, levels: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Convert equi-depth boundaries to free cell masses on a fixed grid.
    grid:(G+1,) edges in [0,1]. Returns (..,G) masses summing to 1."""
    G = grid.shape[0] - 1
    lead = boundaries_x.shape[:-1]
    ge = grid.to(boundaries_x.device).view(*([1] * len(lead)), G + 1).expand(*lead, G + 1)
    cdf_edges = cdf_at(boundaries_x.unsqueeze(-2).expand(*lead, G + 1, boundaries_x.shape[-1]),
                       levels, ge)  # (..,G+1)
    m = (cdf_edges[..., 1:] - cdf_edges[..., :-1]).clamp_min(0.0)
    return m / m.sum(dim=-1, keepdim=True).clamp_min(EPS)


def interval_coverage(grid: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Fractional coverage of each grid cell by [lo,hi]. grid:(G+1,), lo/hi:(..,) -> (..,G)."""
    G = grid.shape[0] - 1
    e0 = grid[:-1].view(*([1] * lo.dim()), G)
    e1 = grid[1:].view(*([1] * lo.dim()), G)
    cell_w = (e1 - e0).clamp_min(EPS)
    inter = (torch.minimum(hi.unsqueeze(-1), e1) - torch.maximum(lo.unsqueeze(-1), e0)).clamp_min(0.0)
    return (inter / cell_w).clamp(0.0, 1.0)  # (..,G) in [0,1]


def ipf_project(m: torch.Tensor, cover: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, n_iter: int = 8) -> torch.Tensor:
    """Differentiable cyclic I-projection of cell masses onto interval-mass constraints.
    m:(B,G) cover:(B,K,G) target:(B,K) mask:(B,K) -> (B,G) projected masses."""
    K = cover.shape[1]
    mp = m
    for _ in range(n_iter):
        for k in range(K):
            A = cover[:, k, :]                      # (B,G)
            t = target[:, k].clamp(EPS, 1 - EPS).unsqueeze(-1)  # (B,1)
            cur = (A * mp).sum(dim=-1, keepdim=True).clamp(EPS, 1 - EPS)  # (B,1)
            scale_in = t / cur
            scale_out = (1 - t) / (1 - cur)
            factor = A * scale_in + (1 - A) * scale_out          # (B,G)
            mk = mask[:, k].unsqueeze(-1)                        # (B,1)
            factor = mk * factor + (1 - mk) * 1.0
            mp = mp * factor
            mp = mp / mp.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return mp


def sel_from_masses(m: torch.Tensor, grid: torch.Tensor, ptype_idx: torch.Tensor,
                    lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Selectivity of predicates from cell masses. m:(B,G) ; ptype/lo/hi:(B,P).
    Uses interval coverage: <= -> [0,v]; >= -> [v,1]; BETWEEN -> [lo,hi]."""
    cov = interval_coverage(grid, lo, hi)              # (B,P,G)
    sel = torch.einsum('bpg,bg->bp', cov, m)           # (B,P)
    return sel.clamp(1e-9, 1.0)


# ───────────────────────── set-encoder model ─────────────────────────

class OasisTorchV3(nn.Module):
    def __init__(self, num_buckets: int = 10, max_obs: int = 16, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 3, ff: int = 128, residual_prior: bool = True):
        super().__init__()
        self.B = num_buckets
        self.K = max_obs
        self.d = d_model
        self.residual_prior = residual_prior
        self.obs_proj = nn.Linear(OBS_FEAT_DIM, d_model)
        self.prior_proj = nn.Linear(3, d_model)      # [boundary_x, level, is_endpoint]
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, batch_first=True,
                                           activation="gelu", dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, ff), nn.GELU(), nn.Linear(ff, num_buckets))

    def forward(self, obs_feats: torch.Tensor, obs_mask: torch.Tensor,
                stale_boundaries: torch.Tensor) -> torch.Tensor:
        """obs_feats:(Bsz,K,OBS_FEAT_DIM) obs_mask:(Bsz,K) stale_boundaries:(Bsz,B+1)
        -> width logits (Bsz,B)."""
        bsz = obs_feats.shape[0]
        levels = torch.linspace(0, 1, self.B + 1, device=obs_feats.device).expand(bsz, self.B + 1)
        is_end = torch.zeros_like(levels); is_end[:, 0] = 1; is_end[:, -1] = 1
        prior_tok = self.prior_proj(torch.stack([stale_boundaries, levels, is_end], dim=-1))
        obs_tok = self.obs_proj(obs_feats)
        cls = self.cls.expand(bsz, 1, self.d)
        tokens = torch.cat([cls, prior_tok, obs_tok], dim=1)   # (Bsz, 1+(B+1)+K, d)
        # key padding mask: True = ignore. CLS+prior always valid; obs by mask.
        valid = torch.cat([torch.ones(bsz, 1 + self.B + 1, device=obs_feats.device),
                           obs_mask], dim=1)
        enc = self.encoder(tokens, src_key_padding_mask=(valid < 0.5))
        logits = self.head(enc[:, 0, :])                       # (Bsz,B)
        if self.residual_prior:
            stale_w = (stale_boundaries[:, 1:] - stale_boundaries[:, :-1]).clamp_min(EPS)
            logits = torch.log(stale_w) + logits               # residual around stale widths
        return logits
