"""
encoder.py — the conditioning encoders.

Three of them, feeding two different attention paths:

  ProgramEncoder   room graph (GATv2)      -> (N, d)   static memory
  BoundaryEncoder  footprint + entrance    -> (65, d)  static memory
  StateEncoder     per-step occupancy      -> (N, S, d) PER-STEP memory  [R1]

The first two are v2's, extended with Vastu embeddings (R4). The third is
new and is the point of v3: the decoder gets a fresh look at the board before
every single room instead of a memory of the last one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.placer_v3.config import PlacerV3Config


# ── graph attention ─────────────────────────────────────────────────────────

class GATv2Layer(nn.Module):
    """Single-graph GATv2 (Brody et al. 2021): the nonlinearity precedes the
    attention projection, so attention is genuinely input-dependent."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0, "node_dim must divide by gnn_heads"
        self.heads = heads
        self.hd = dim // heads
        self.lin_l = nn.Linear(dim, dim)
        self.lin_r = nn.Linear(dim, dim)
        self.att = nn.Parameter(torch.empty(heads, self.hd))
        self.leaky = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor
                ) -> torch.Tensor:
        n = x.size(0)
        xl = self.lin_l(x).view(n, self.heads, self.hd)
        xr = self.lin_r(x).view(n, self.heads, self.hd)
        src, dst = edge_index[0], edge_index[1]
        e = self.leaky(xl[src] + xr[dst])
        scores = (e * self.att).sum(dim=-1)
        scores = scores - _seg_max(scores, dst, n)[dst]
        alpha = scores.exp()
        denom = _seg_sum(alpha, dst, n).clamp_min(1e-16)
        alpha = self.drop(alpha / denom[dst])
        msg = xl[src] * alpha.unsqueeze(-1)
        out = _seg_sum(msg.reshape(msg.size(0), -1), dst, n)
        return out.view(n, self.heads * self.hd)


def _seg_sum(vals: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    out = vals.new_zeros((n,) + vals.shape[1:])
    return out.index_add_(0, idx, vals)


def _seg_max(vals: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    out = vals.new_full((n,) + vals.shape[1:], float("-inf"))
    out = out.index_reduce_(0, idx, vals, "amax", include_self=True)
    return torch.nan_to_num(out, neginf=0.0)


class ProgramEncoder(nn.Module):
    """Room graph -> per-room memory tokens.

    Vastu rides on the NODE, not the global vector, because it is a per-room
    constraint: "the pooja room wants NE" says nothing about the kitchen.
    """

    def __init__(self, cfg: PlacerV3Config):
        super().__init__()
        d = cfg.node_dim
        self.type_emb = nn.Embedding(cfg.n_room_types, d, padding_idx=0)
        self.zone_emb = nn.Embedding(cfg.n_zones, d, padding_idx=0)
        self.floor_emb = nn.Embedding(8, d)
        self.vastu_dir_emb = nn.Embedding(cfg.n_vastu_dirs, d, padding_idx=0)
        self.vastu_str_emb = nn.Embedding(cfg.n_vastu_strengths, d,
                                          padding_idx=0)
        self.layers = nn.ModuleList(
            [GATv2Layer(d, cfg.gnn_heads, cfg.dropout)
             for _ in range(cfg.gnn_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(cfg.gnn_layers)])
        self.out = nn.Linear(d, cfg.d_model)

    def forward(self, type_ids, zone_ids, floor_ids, vastu_dir_ids,
                vastu_strength_ids, edge_index):
        n = type_ids.size(0)
        h = (self.type_emb(type_ids)
             + self.zone_emb(zone_ids)
             + self.floor_emb(floor_ids.clamp(0, 7))
             + self.vastu_dir_emb(vastu_dir_ids)
             + self.vastu_str_emb(vastu_strength_ids))
        # self-loops keep isolated nodes' own signal alive
        loops = torch.arange(n, device=h.device).unsqueeze(0).repeat(2, 1)
        ei = torch.cat([edge_index, loops], dim=1) if edge_index.numel() \
            else loops
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.elu(layer(h, ei)))
        return self.out(h)


class BoundaryEncoder(nn.Module):
    """(2,64,64) footprint -> 64 spatial tokens, plus one global token."""

    def __init__(self, cfg: PlacerV3Config):
        super().__init__()
        c = cfg.cnn_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(2, c // 4, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c // 4, c // 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c // 2, c, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c, c, 3, stride=1, padding=1), nn.ReLU(),
        )
        self.proj = nn.Linear(c, cfg.d_model)
        self.global_mlp = nn.Sequential(
            nn.Linear(cfg.global_dim, cfg.d_model), nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model))

    def forward(self, boundary, glob):
        feat = self.cnn(boundary.unsqueeze(0))
        tokens = self.proj(feat.flatten(2).transpose(1, 2).squeeze(0))
        gtok = self.global_mlp(glob).unsqueeze(0)
        return torch.cat([tokens, gtok], dim=0)


class StateEncoder(nn.Module):
    """R1 — (N, C, 32, 32) occupancy stack -> (N, state_tokens, d_model).

    Every decode step gets its own small set of memory tokens describing the
    board as it stood before that room was placed. The whole stack is encoded
    in ONE batched conv pass (batch dim = the step), which is why per-step
    state costs almost nothing at N <= 24 rooms.

    A learned per-token position embedding is added so the decoder can tell
    "top-left of the plot is full" from "bottom-right is full" — without it
    the 4x4 pooled grid would be an unordered bag.
    """

    def __init__(self, cfg: PlacerV3Config):
        super().__init__()
        c = cfg.state_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(cfg.state_channels, c // 2, 3, stride=2, padding=1),
            nn.ReLU(),                                             # 16
            nn.Conv2d(c // 2, c, 3, stride=2, padding=1), nn.ReLU(),   # 8
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.ReLU(),        # 4
        )
        self.proj = nn.Linear(c, cfg.d_model)
        self.pos = nn.Parameter(
            torch.zeros(cfg.state_tokens, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # state: (N, C, 32, 32) -> (N, 16, d_model)
        feat = self.cnn(state)                       # (N, c, 4, 4)
        tokens = feat.flatten(2).transpose(1, 2)     # (N, 16, c)
        return self.proj(tokens) + self.pos
