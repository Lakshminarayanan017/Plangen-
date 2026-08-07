"""
decoder.py — the autoregressive placement decoder (Tier 2 core).

One decoder step per room, in canonical generation order. Step i attends
causally to rooms 0..i and cross-attends to the encoder memory (program
nodes + boundary tokens + global). Its input embeds room i's identity
(type/zone) plus the PLACEMENT of room i-1 (its cell + size) — the standard
autoregressive teacher-forcing signal, so the model conditions on where prior
rooms went. Two heads read each step's hidden state: a cell head (1024
logits) and a size head (40 logits).

At inference the caller applies decode-time legality masks to the cell logits
before sampling (tier2_placer.masked_decode) — the mechanism that makes an
invalid layout unsamplable, the property v1 lacked.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.tier2_placer.config import PlacerConfig

# special "previous placement" tokens for the first room step
_START_CELL = 0            # reuse index space; disambiguated by a start flag
_START_SIZE = 0


class DecoderLayer(nn.Module):
    def __init__(self, cfg: PlacerConfig):
        super().__init__()
        d = cfg.d_model
        self.self_attn = nn.MultiheadAttention(
            d, cfg.dec_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d, cfg.dec_heads, dropout=cfg.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg.ff_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ff_dim, d))
        self.n1, self.n2, self.n3 = (nn.LayerNorm(d) for _ in range(3))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, causal_mask):
        h = self.n1(x)
        x = x + self.drop(self.self_attn(
            h, h, h, attn_mask=causal_mask, need_weights=False)[0])
        h = self.n2(x)
        x = x + self.drop(self.cross_attn(
            h, memory, memory, need_weights=False)[0])
        x = x + self.drop(self.ff(self.n3(x)))
        return x


class PlacementDecoder(nn.Module):
    def __init__(self, cfg: PlacerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.type_emb = nn.Embedding(cfg.n_room_types, d, padding_idx=0)
        self.zone_emb = nn.Embedding(cfg.n_zones, d, padding_idx=0)
        # embeddings of the PREVIOUS room's placement (+1 slot for START)
        self.prev_cell_emb = nn.Embedding(cfg.cell_count + 1, d)
        self.prev_size_emb = nn.Embedding(cfg.size_count + 1, d)
        self.pos_emb = nn.Embedding(cfg.max_rooms, d)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(cfg.dec_layers)])
        self.norm = nn.LayerNorm(d)
        self.cell_head = nn.Linear(d, cfg.cell_count)
        self.size_head = nn.Linear(d, cfg.size_count)

    def step_input(self, type_ids, zone_ids, prev_cell, prev_size):
        """Embed one full room sequence's decoder inputs (N, d)."""
        n = type_ids.size(0)
        pos = torch.arange(n, device=type_ids.device)
        return (self.type_emb(type_ids) + self.zone_emb(zone_ids)
                + self.prev_cell_emb(prev_cell) + self.prev_size_emb(prev_size)
                + self.pos_emb(pos.clamp_max(self.cfg.max_rooms - 1)))

    def forward(self, x, memory):
        """x: (N, d) decoder inputs; memory: (M, d). Returns (cell_logits
        (N, 1024), size_logits (N, 40))."""
        n = x.size(0)
        causal = torch.triu(
            torch.full((n, n), float("-inf"), device=x.device), diagonal=1)
        x = x.unsqueeze(0); mem = memory.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mem, causal)
        h = self.norm(x).squeeze(0)
        return self.cell_head(h), self.size_head(h)


def shift_prev(target_cell: torch.Tensor, target_size: torch.Tensor,
               cell_count: int, size_count: int):
    """Teacher-forcing inputs: prev placement per step = the previous room's
    (cell, size); step 0 gets the START slot (index == count)."""
    n = target_cell.size(0)
    prev_cell = torch.full((n,), cell_count, dtype=torch.long,
                           device=target_cell.device)
    prev_size = torch.full((n,), size_count, dtype=torch.long,
                           device=target_size.device)
    if n > 1:
        prev_cell[1:] = target_cell[:-1]
        prev_size[1:] = target_size[:-1]
    return prev_cell, prev_size
