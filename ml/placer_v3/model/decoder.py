"""
decoder.py — the autoregressive placement decoder.

One step per room, in canonical generation order. Each layer runs three
attentions instead of v2's two:

  1. causal self-attention over rooms 0..i        (what has been decided)
  2. cross-attention to STATIC memory             (program graph + footprint)
  3. cross-attention to THIS STEP's state tokens  (what the board looks like)

(3) is R1. It is batched over the step dimension — query (N,1,d) against
key/value (N,S,d) — so step i can only ever attend to its own state slice.
Causality is structural, not a mask that could be got wrong.

Five heads read each step: cell and size as in v2, plus aspect, orientation
and band (R3), which give the carver proportion and structure instead of
making it derive everything from a single seed cell.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ml.placer_v3.config import PlacerV3Config


class DecoderLayer(nn.Module):
    def __init__(self, cfg: PlacerV3Config):
        super().__init__()
        d = cfg.d_model
        self.self_attn = nn.MultiheadAttention(
            d, cfg.dec_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(
            d, cfg.dec_heads, dropout=cfg.dropout, batch_first=True)
        self.state_attn = nn.MultiheadAttention(
            d, cfg.dec_heads, dropout=cfg.dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d, cfg.ff_dim), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.ff_dim, d))
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d)
                                              for _ in range(4))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, state_tokens, causal_mask):
        # x: (1, N, d) · memory: (1, M, d) · state_tokens: (N, S, d)
        h = self.n1(x)
        x = x + self.drop(self.self_attn(
            h, h, h, attn_mask=causal_mask, need_weights=False)[0])

        h = self.n2(x)
        x = x + self.drop(self.cross_attn(
            h, memory, memory, need_weights=False)[0])

        # per-step state: batch over the sequence so query i sees only its
        # own board snapshot. (1,N,d) -> (N,1,d) -> attend -> (1,N,d).
        h = self.n3(x).transpose(0, 1)                     # (N, 1, d)
        s = self.state_attn(h, state_tokens, state_tokens,
                            need_weights=False)[0]         # (N, 1, d)
        x = x + self.drop(s.transpose(0, 1))               # (1, N, d)

        return x + self.drop(self.ff(self.n4(x)))


class PlacementDecoder(nn.Module):
    def __init__(self, cfg: PlacerV3Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.type_emb = nn.Embedding(cfg.n_room_types, d, padding_idx=0)
        self.zone_emb = nn.Embedding(cfg.n_zones, d, padding_idx=0)
        self.vastu_dir_emb = nn.Embedding(cfg.n_vastu_dirs, d, padding_idx=0)
        # previous room's realised placement (+1 index for the START slot)
        self.prev_cell_emb = nn.Embedding(cfg.cell_count + 1, d)
        self.prev_size_emb = nn.Embedding(cfg.size_count + 1, d)
        self.pos_emb = nn.Embedding(cfg.max_rooms, d)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(cfg.dec_layers)])
        self.norm = nn.LayerNorm(d)

        self.cell_head = nn.Linear(d, cfg.cell_count)
        self.size_head = nn.Linear(d, cfg.size_count)
        self.aspect_head = nn.Linear(d, cfg.aspect_count)
        self.orientation_head = nn.Linear(d, cfg.orientation_count)
        self.band_head = nn.Linear(d, cfg.band_count)

    def step_input(self, type_ids, zone_ids, vastu_dir_ids,
                   prev_cell, prev_size) -> torch.Tensor:
        n = type_ids.size(0)
        pos = torch.arange(n, device=type_ids.device)
        return (self.type_emb(type_ids)
                + self.zone_emb(zone_ids)
                + self.vastu_dir_emb(vastu_dir_ids)
                + self.prev_cell_emb(prev_cell)
                + self.prev_size_emb(prev_size)
                + self.pos_emb(pos.clamp_max(self.cfg.max_rooms - 1)))

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                state_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (N, d) · memory: (M, d) · state_tokens: (N, S, d)."""
        n = x.size(0)
        causal = torch.triu(
            torch.full((n, n), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1)
        x = x.unsqueeze(0)
        mem = memory.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, mem, state_tokens, causal)
        h = self.norm(x).squeeze(0)
        return {
            "cell": self.cell_head(h),
            "size": self.size_head(h),
            "aspect": self.aspect_head(h),
            "orientation": self.orientation_head(h),
            "band": self.band_head(h),
        }


def shift_prev(target_cell: torch.Tensor, target_size: torch.Tensor,
               cell_count: int, size_count: int
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forcing inputs: step i is told room i-1's realised placement;
    step 0 gets the dedicated START index (== count)."""
    n = target_cell.size(0)
    prev_cell = torch.full((n,), cell_count, dtype=torch.long,
                           device=target_cell.device)
    prev_size = torch.full((n,), size_count, dtype=torch.long,
                           device=target_size.device)
    if n > 1:
        prev_cell[1:] = target_cell[:-1]
        prev_size[1:] = target_size[:-1]
    return prev_cell, prev_size
