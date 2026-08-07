"""
encoder.py — the conditioning encoders (Tier 0).

ProgramEncoder: a GATv2-style attention GNN over the room graph → per-room
embeddings. BoundaryEncoder: a small CNN over the 2-channel 64×64 boundary
raster → 8×8 spatial memory tokens. A global MLP folds plot-context scalars
into a single memory token. Together they form the cross-attention memory the
decoder attends to.

Batching note: variable room counts per plan are handled by processing ONE
plan at a time (batch dim = 1 in the graph ops) and padding the decoder side.
The GNN here therefore takes a single graph (N nodes) — the training loop
loops plans within a minibatch and averages loss. This keeps the graph code
dependency-free (no torch-geometric) and is fast enough at this scale.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.tier2_placer.config import GLOBAL_DIM, PlacerConfig


class GATv2Layer(nn.Module):
    """Single-graph GATv2 attention layer (Brody et al. 2021 formulation:
    the nonlinearity precedes the attention projection, so attention is
    genuinely input-dependent). Multi-head, mean-aggregated."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0
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
        # x: (N, dim); edge_index: (2, E) with self-loops added by caller
        n = x.size(0)
        xl = self.lin_l(x).view(n, self.heads, self.hd)   # (N,H,hd) source
        xr = self.lin_r(x).view(n, self.heads, self.hd)   # (N,H,hd) target
        src, dst = edge_index[0], edge_index[1]
        # GATv2: score = a · LeakyReLU(x_src_l + x_dst_r)
        e = self.leaky(xl[src] + xr[dst])                 # (E,H,hd)
        scores = (e * self.att).sum(dim=-1)               # (E,H)
        # softmax over incoming edges per destination node, per head
        scores = scores - _seg_max(scores, dst, n)[dst]
        alpha = scores.exp()
        denom = _seg_sum(alpha, dst, n).clamp_min(1e-16)  # (N,H)
        alpha = self.drop(alpha / denom[dst])             # (E,H)
        msg = xl[src] * alpha.unsqueeze(-1)               # (E,H,hd)
        out = _seg_sum(msg.reshape(msg.size(0), -1), dst, n)  # (N, dim)
        return out.view(n, self.heads * self.hd)


def _seg_sum(vals: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    out = vals.new_zeros((n,) + vals.shape[1:])
    return out.index_add_(0, idx, vals)


def _seg_max(vals: torch.Tensor, idx: torch.Tensor, n: int) -> torch.Tensor:
    out = vals.new_full((n,) + vals.shape[1:], float("-inf"))
    out = out.index_reduce_(0, idx, vals, "amax", include_self=True)
    # nodes with no incoming edge stay -inf → make them 0 to avoid nan
    return torch.nan_to_num(out, neginf=0.0)


class ProgramEncoder(nn.Module):
    def __init__(self, cfg: PlacerConfig):
        super().__init__()
        d = cfg.node_dim
        self.type_emb = nn.Embedding(cfg.n_room_types, d, padding_idx=0)
        self.zone_emb = nn.Embedding(cfg.n_zones, d, padding_idx=0)
        self.floor_emb = nn.Embedding(8, d)
        self.layers = nn.ModuleList(
            [GATv2Layer(d, cfg.gnn_heads, cfg.dropout)
             for _ in range(cfg.gnn_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(d) for _ in range(cfg.gnn_layers)])
        self.out = nn.Linear(d, cfg.d_model)

    def forward(self, type_ids, zone_ids, floor_ids, edge_index):
        n = type_ids.size(0)
        h = (self.type_emb(type_ids) + self.zone_emb(zone_ids)
             + self.floor_emb(floor_ids.clamp(0, 7)))
        # add self-loops so isolated nodes keep their own signal
        loops = torch.arange(n, device=h.device).unsqueeze(0).repeat(2, 1)
        ei = torch.cat([edge_index, loops], dim=1) if edge_index.numel() \
            else loops
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + F.elu(layer(h, ei)))
        return self.out(h)               # (N, d_model) program memory tokens


class BoundaryEncoder(nn.Module):
    """2×64×64 -> 8×8×cnn_dim -> (64, d_model) memory tokens + a global token
    from plot-context scalars."""

    def __init__(self, cfg: PlacerConfig):
        super().__init__()
        c = cfg.cnn_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(2, c // 4, 3, stride=2, padding=1), nn.ReLU(),   # 32
            nn.Conv2d(c // 4, c // 2, 3, stride=2, padding=1), nn.ReLU(),  # 16
            nn.Conv2d(c // 2, c, 3, stride=2, padding=1), nn.ReLU(),   # 8
            nn.Conv2d(c, c, 3, stride=1, padding=1), nn.ReLU(),        # 8
        )
        self.proj = nn.Linear(c, cfg.d_model)
        self.global_mlp = nn.Sequential(
            nn.Linear(GLOBAL_DIM, cfg.d_model),
            nn.ReLU(), nn.Linear(cfg.d_model, cfg.d_model))

    def forward(self, boundary, glob):
        # boundary: (2,64,64); glob: (G,)
        feat = self.cnn(boundary.unsqueeze(0))             # (1,c,8,8)
        tokens = feat.flatten(2).transpose(1, 2).squeeze(0)  # (64, c)
        tokens = self.proj(tokens)                         # (64, d_model)
        gtok = self.global_mlp(glob).unsqueeze(0)          # (1, d_model)
        return torch.cat([tokens, gtok], dim=0)            # (65, d_model)
